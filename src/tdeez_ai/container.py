"""Dependency wiring: picks mock or real AWS implementations from settings."""

from dataclasses import dataclass

import boto3

from tdeez_ai.bedrock.canvas import ImageGenerator, NovaCanvasGenerator
from tdeez_ai.bedrock.client import BedrockClient
from tdeez_ai.bedrock.embeddings import ImageEmbedder, TitanEmbedder
from tdeez_ai.bedrock.guard import CostGuard
from tdeez_ai.bedrock.mock import MockEmbedder, MockImageGenerator, MockVisionAnalyzer
from tdeez_ai.bedrock.vision import NovaVisionAnalyzer, VisionAnalyzer
from tdeez_ai.core.auth import CognitoJwtVerifier, MockTokenVerifier, TokenVerifier
from tdeez_ai.core.config import Settings
from tdeez_ai.repositories import dynamodb as ddb
from tdeez_ai.repositories.base import (
    CreationRepository,
    ImageStore,
    Mailer,
    ProfileRepository,
    QuoteRepository,
    SimilarityIndex,
    SlabRepository,
)
from tdeez_ai.repositories.memory import (
    InMemoryCreationRepository,
    InMemoryImageStore,
    InMemoryMailer,
    InMemoryProfileRepository,
    InMemoryQuoteRepository,
    InMemorySimilarityIndex,
    InMemorySlabRepository,
)
from tdeez_ai.repositories.opensearch import (
    OpenSearchSimilarityIndex,
    create_client,
    ensure_index,
)
from tdeez_ai.repositories.s3 import S3ImageStore
from tdeez_ai.repositories.ses import SesMailer
from tdeez_ai.seed import seed_inventory
from tdeez_ai.services.concepts import ConceptService
from tdeez_ai.services.creations import CreationService
from tdeez_ai.services.customization import CustomizationService
from tdeez_ai.services.image_access import ImageAccess
from tdeez_ai.services.pricing import PricingService
from tdeez_ai.services.profile import ProfileService
from tdeez_ai.services.quotes import QuoteService
from tdeez_ai.services.room import RoomService
from tdeez_ai.services.similarity import SimilarityService
from tdeez_ai.services.slab_analysis import SlabAnalysisService, SlabIsolator
from tdeez_ai.services.slab_source import SlabSourceService


@dataclass
class Container:
    settings: Settings
    verifier: TokenVerifier
    store: ImageStore
    slabs: SlabRepository
    creation_repo: CreationRepository
    quote_repo: QuoteRepository
    profile_repo: ProfileRepository
    index: SimilarityIndex
    embedder: ImageEmbedder
    mailer: Mailer
    slab_source: SlabSourceService
    similarity: SimilarityService
    analysis: SlabAnalysisService
    concepts: ConceptService
    rooms: RoomService
    customization: CustomizationService
    creations: CreationService
    quotes: QuoteService
    profiles: ProfileService


def build_container(
    settings: Settings,
    *,
    vision: VisionAnalyzer | None = None,
    images: ImageGenerator | None = None,
    embedder: ImageEmbedder | None = None,
    verifier: TokenVerifier | None = None,
    mailer: Mailer | None = None,
    store: ImageStore | None = None,
) -> Container:
    """Wire everything. Keyword overrides let tests inject fakes for any single dependency."""
    guard = CostGuard(settings.max_images_per_request, settings.max_image_bytes)
    slabs: SlabRepository
    creation_repo: CreationRepository
    quote_repo: QuoteRepository
    profile_repo: ProfileRepository
    index: SimilarityIndex

    if settings.use_mock:
        slabs = InMemorySlabRepository()
        creation_repo = InMemoryCreationRepository()
        quote_repo = InMemoryQuoteRepository()
        profile_repo = InMemoryProfileRepository()
        index = InMemorySimilarityIndex()
        store = store or InMemoryImageStore(settings.mock_public_base_url)
        mailer = mailer or InMemoryMailer()
        images = images or MockImageGenerator()
        vision = vision or MockVisionAnalyzer()
        embedder = embedder or MockEmbedder(settings.embedding_dimensions)
        verifier = verifier or MockTokenVerifier()
    else:
        region = settings.aws_region
        dynamo = boto3.resource("dynamodb", region_name=region)
        assert settings.slabs_table and settings.creations_table  # guaranteed by Settings
        assert settings.profiles_table and settings.quotes_table
        assert settings.s3_bucket and settings.opensearch_endpoint and settings.ses_sender
        assert settings.cognito_user_pool_id and settings.cognito_app_client_id
        slabs = ddb.DynamoSlabRepository(dynamo.Table(settings.slabs_table))
        creation_repo = ddb.DynamoCreationRepository(dynamo.Table(settings.creations_table))
        quote_repo = ddb.DynamoQuoteRepository(dynamo.Table(settings.quotes_table))
        profile_repo = ddb.DynamoProfileRepository(dynamo.Table(settings.profiles_table))
        os_client = create_client(settings.opensearch_endpoint, region)
        ensure_index(os_client, settings.opensearch_index, settings.embedding_dimensions)
        index = OpenSearchSimilarityIndex(os_client, settings.opensearch_index)
        store = store or S3ImageStore(boto3.client("s3", region_name=region), settings.s3_bucket)
        mailer = mailer or SesMailer(boto3.client("sesv2", region_name=region), settings.ses_sender)
        bedrock = BedrockClient.from_settings(settings)
        images = images or NovaCanvasGenerator(bedrock, settings.canvas_model_id, guard)
        vision = vision or NovaVisionAnalyzer(bedrock, settings.vision_model_id, guard)
        embedder = embedder or TitanEmbedder(
            bedrock, settings.embedding_model_id, settings.embedding_dimensions, guard
        )
        verifier = verifier or CognitoJwtVerifier(
            settings.aws_region, settings.cognito_user_pool_id, settings.cognito_app_client_id
        )

    access = ImageAccess(store, settings)
    isolator = SlabIsolator(access, images, profile_repo)
    slab_source = SlabSourceService(store, slabs, profile_repo, access, settings)

    if settings.use_mock and settings.seed_on_startup:
        seed_inventory(slabs, store, embedder, index)

    return Container(
        settings=settings,
        verifier=verifier,
        store=store,
        slabs=slabs,
        creation_repo=creation_repo,
        quote_repo=quote_repo,
        profile_repo=profile_repo,
        index=index,
        embedder=embedder,
        mailer=mailer,
        slab_source=slab_source,
        similarity=SimilarityService(access, embedder, index, slabs, slab_source),
        analysis=SlabAnalysisService(access, isolator, vision, slabs, profile_repo, settings),
        concepts=ConceptService(access, isolator, images, profile_repo, settings),
        rooms=RoomService(access, images, vision, settings),
        customization=CustomizationService(
            access, images, vision, PricingService.from_settings(settings)
        ),
        creations=CreationService(creation_repo, slabs, access),
        quotes=QuoteService(quote_repo, creation_repo, slabs, mailer, settings),
        profiles=ProfileService(profile_repo),
    )

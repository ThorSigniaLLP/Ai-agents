from fastapi_admin.app import app as admin_app
from fastapi_admin.resources import Model
from core.models import ResearchJob, SearchCandidate, FetchedPage, EvidenceChunkModel, ExtractedItem, NodeLog, PipelineError

@admin_app.register
class ResearchJobResource(Model):
    label = "Research Jobs"
    model = ResearchJob
    icon = "fas fa-briefcase"

@admin_app.register
class SearchCandidateResource(Model):
    label = "Search Candidates"
    model = SearchCandidate
    icon = "fas fa-search"

@admin_app.register
class FetchedPageResource(Model):
    label = "Fetched Pages"
    model = FetchedPage
    icon = "fas fa-file-alt"

@admin_app.register
class EvidenceChunkResource(Model):
    label = "Evidence Chunks"
    model = EvidenceChunkModel
    icon = "fas fa-puzzle-piece"

@admin_app.register
class ExtractedItemResource(Model):
    label = "Extracted Items"
    model = ExtractedItem
    icon = "fas fa-list"

@admin_app.register
class NodeLogResource(Model):
    label = "Node Logs"
    model = NodeLog
    icon = "fas fa-clock"

@admin_app.register
class PipelineErrorResource(Model):
    label = "Pipeline Errors"
    model = PipelineError
    icon = "fas fa-exclamation-triangle"

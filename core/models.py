from tortoise import fields
from tortoise.models import Model

class ResearchJob(Model):
    job_id = fields.CharField(pk=True, max_length=255)
    company_name = fields.CharField(max_length=255)
    status = fields.CharField(max_length=50, default="started")
    created_at = fields.DatetimeField(auto_now_add=True)
    completed_at = fields.DatetimeField(null=True)
    metadata_json = fields.JSONField(default=dict)

    class Meta:
        table = "research_jobs"


class NodeLog(Model):
    id = fields.IntField(pk=True)
    job = fields.ForeignKeyField("models.ResearchJob", related_name="node_logs")
    node_name = fields.CharField(max_length=255)
    duration_seconds = fields.FloatField(default=0.0)
    created_at = fields.DatetimeField(auto_now_add=True)

    class Meta:
        table = "node_logs"


class SearchCandidate(Model):
    id = fields.IntField(pk=True)
    job = fields.ForeignKeyField("models.ResearchJob", related_name="candidates")
    url = fields.TextField()
    domain = fields.CharField(max_length=255, null=True)
    title = fields.TextField(null=True)
    provider = fields.CharField(max_length=50, null=True)
    domain_score = fields.FloatField(null=True)
    created_at = fields.DatetimeField(auto_now_add=True)

    class Meta:
        table = "search_candidates"


class FetchedPage(Model):
    id = fields.IntField(pk=True)
    job = fields.ForeignKeyField("models.ResearchJob", related_name="pages")
    url = fields.TextField()
    title = fields.TextField(null=True)
    content_text = fields.TextField(null=True)
    fetch_method = fields.CharField(max_length=50, null=True)
    status_code = fields.IntField(null=True)
    created_at = fields.DatetimeField(auto_now_add=True)

    class Meta:
        table = "fetched_pages"


class EvidenceChunkModel(Model):
    id = fields.IntField(pk=True)
    job = fields.ForeignKeyField("models.ResearchJob", related_name="chunks")
    url = fields.TextField()
    chunk_index = fields.IntField()
    chunk_text = fields.TextField(null=True)
    rerank_score = fields.FloatField(null=True)
    created_at = fields.DatetimeField(auto_now_add=True)

    class Meta:
        table = "evidence_chunks"


class ExtractedItem(Model):
    id = fields.IntField(pk=True)
    job = fields.ForeignKeyField("models.ResearchJob", related_name="extracted_items")
    field = fields.CharField(max_length=255)
    value = fields.TextField(null=True)
    source_url = fields.TextField(null=True)
    created_at = fields.DatetimeField(auto_now_add=True)

    class Meta:
        table = "extracted_items"


class PipelineError(Model):
    id = fields.IntField(pk=True)
    job = fields.ForeignKeyField("models.ResearchJob", related_name="errors")
    node_name = fields.CharField(max_length=255, null=True)
    error_message = fields.TextField(null=True)
    created_at = fields.DatetimeField(auto_now_add=True)

    class Meta:
        table = "pipeline_errors"

class Admin(Model):
    id = fields.IntField(pk=True)
    username = fields.CharField(max_length=50, unique=True)
    password = fields.CharField(max_length=200)
    is_active = fields.BooleanField(default=True)
    is_superuser = fields.BooleanField(default=True)
    avatar = fields.CharField(max_length=200, default="")
    intro = fields.TextField(default="")
    created_at = fields.DatetimeField(auto_now_add=True)

    def __str__(self):
        return self.username



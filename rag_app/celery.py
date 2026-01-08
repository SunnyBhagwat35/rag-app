import os

from celery import Celery

# Set the default Django settings module for the 'celery' program.
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'rag_app.settings')

app = Celery('rag_app')

# Using a string here means the worker doesn't have to serialize
# the configuration object to child processes.
# - namespace='CELERY' means all celery-related configuration keys
#   should have a `CELERY_` prefix.
app.config_from_object('django.conf:settings', namespace='CELERY')

# Load task modules from all registered Django apps.
app.autodiscover_tasks()

app.conf.update(
    task_routes = {
        'aichat.tasks.process_documents': {'queue': 'embedding_queue'}, 
    },
    task_create_missing_queues = True,
    task_time_limit=1800,
    task_soft_time_limit=1740,
    task_acks_late=True,
)
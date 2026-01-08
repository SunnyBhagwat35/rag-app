from django.apps import AppConfig


class DocumentStoreConfig(AppConfig):
    name = 'document_store'

    def ready(self):
        import document_store.signals
        
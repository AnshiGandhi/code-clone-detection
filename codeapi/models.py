from django.db import models

# Create your models here.
class snippet(models.Model):
    snippet = models.CharField(max_length = 10000)

    def __str__(self):
        return self.snippet


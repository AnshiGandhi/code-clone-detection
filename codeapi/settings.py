INSTALLED_APPS = [
    'corsheaders'
] 
MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',  # Add this
    'django.middleware.common.CommonMiddleware'
]
CORS_ALLOW_ALL_ORIGINS = True  # Previously CORS_ORIGIN_ALLOW_ALL in older versions



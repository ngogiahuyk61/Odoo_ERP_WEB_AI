import os
from pathlib import Path

# Đường dẫn gốc
BASE_DIR = Path(__file__).resolve().parent.parent

# SECRET_KEY (phải luôn có, không để rỗng)
SECRET_KEY = "%z1%450a$-oz(*tq#669j(^^4$$xkjih_jt4^#m)ht2j^)u-xy"

# Debug mode
DEBUG = True

# Khi DEBUG=True thì ALLOWED_HOSTS có thể để rỗng
ALLOWED_HOSTS = [
    'localhost',
    '127.0.0.1',
    'odoo-erp-web-ai.onrender.com',
]

# Ứng dụng cài đặt
INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "floorplan_app",  # app chính
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "floorplan_project.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],  # để HTML ngoài app
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "floorplan_project.wsgi.application"

# Database mặc định (SQLite cho dev)
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

# Ngôn ngữ và múi giờ
LANGUAGE_CODE = "en-us"
TIME_ZONE = "Asia/Ho_Chi_Minh"
USE_I18N = True
USE_TZ = True

# Static files
STATIC_URL = "/static/"

STATICFILES_DIRS = [
    BASE_DIR / "floorplan_app" / "static",  # chỗ chứa CSS, JS, img
]

STATIC_ROOT = BASE_DIR / "staticfiles"  # nơi collectstatic gom file

# Media (cho file upload sau này)
MEDIA_URL = "/media/"
MEDIA_ROOT = BASE_DIR / "media"

# Default primary key
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

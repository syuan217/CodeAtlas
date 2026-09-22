# java-mini

Fixture repository for indexer tests.

## Model Layer

`com.example.model.User` holds the user entity with two fields.

## Service Layer

`com.example.service.UserService` validates and saves users.

The service has one deliberately long method to exercise the
word-level fallback chunker path.

## Ignored Files

Files under `target/` and `*.log` are excluded by `.gitignore`.

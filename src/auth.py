import logging
from functools import lru_cache

import boto3
import jwt
from jwt import PyJWKClient

from config import COGNITO_APP_CLIENT_ID, COGNITO_REGION, COGNITO_USER_POOL_ID, JWT_ISSUER, JWT_JWKS_URL


LOGGER = logging.getLogger(__name__)


class AuthenticationError(Exception):
    pass


def extract_bearer_token(event):
    headers = (event or {}).get("headers") or {}
    authorization = headers.get("authorization") or headers.get("Authorization") or ""
    if not authorization:
        raise AuthenticationError("Missing Authorization header.")

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise AuthenticationError("Authorization header must use Bearer token format.")
    return token.strip()


def extract_jwt_claims(token):
    claims = _decode_jwt(token)
    token_use = (claims.get("token_use") or "").strip().lower()
    if token_use not in {"access", "id"}:
        raise ValueError("Unsupported token_use in JWT.")

    expected_client_id = (COGNITO_APP_CLIENT_ID or "").strip()
    if expected_client_id:
        if token_use == "access":
            actual_client_id = str(claims.get("client_id") or "").strip()
            if actual_client_id != expected_client_id:
                raise ValueError("JWT client_id does not match the configured Cognito app client.")
        elif token_use == "id":
            actual_audience = str(claims.get("aud") or "").strip()
            if actual_audience != expected_client_id:
                raise ValueError("JWT aud does not match the configured Cognito app client.")

    return claims


def validate_access_token(token):
    claims = extract_jwt_claims(token)
    return {"claims": claims}


def get_user_email_from_token(token):
    """Call Cognito GetUser API using the access token to retrieve the verified email."""
    try:
        client = _cognito_client()
        response = client.get_user(AccessToken=token)
        attributes = {attr["Name"]: attr["Value"] for attr in response.get("UserAttributes", [])}
        email = attributes.get("email", "").strip()
        if not email:
            raise AuthenticationError("Email attribute not found in Cognito user profile.")
        LOGGER.info("Retrieved email from Cognito for user: %s", response.get("Username"))
        return email
    except AuthenticationError:
        raise
    except Exception as error:
        LOGGER.warning("Failed to retrieve user email from Cognito: %s", error)
        raise AuthenticationError("Failed to retrieve user profile from Cognito.") from error


def _decode_jwt(token):
    issuer = _issuer()
    jwks_client = _jwks_client()

    try:
        signing_key = jwks_client.get_signing_key_from_jwt(token)
        options = {"require": ["exp", "iat", "iss", "token_use"], "verify_aud": False}
        kwargs = {
            "algorithms": ["RS256"],
            "issuer": issuer,
            "options": options,
        }
        return jwt.decode(token, signing_key.key, **kwargs)
    except Exception as error:
        LOGGER.warning("JWT validation failed: %s", error)
        raise ValueError("Invalid or expired access token.") from error


@lru_cache(maxsize=1)
def _jwks_client():
    return PyJWKClient(_jwks_url())


@lru_cache(maxsize=1)
def _cognito_client():
    return boto3.client("cognito-idp", region_name=COGNITO_REGION)


def _issuer():
    if JWT_ISSUER:
        return JWT_ISSUER
    if not COGNITO_USER_POOL_ID:
        raise AuthenticationError("COGNITO_USER_POOL_ID is not configured.")
    return f"https://cognito-idp.{COGNITO_REGION}.amazonaws.com/{COGNITO_USER_POOL_ID}"


def _jwks_url():
    if JWT_JWKS_URL:
        return JWT_JWKS_URL
    return f"{_issuer()}/.well-known/jwks.json"

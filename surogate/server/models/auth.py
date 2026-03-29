"""
Pydantic schemas for Authentication API
"""

from pydantic import BaseModel, Field

class AuthLoginRequest(BaseModel):
    """Login payload: username/password to obtain a JWT."""

    username: str = Field(..., description = "Username")
    password: str = Field(..., description = "Password")
    
class RefreshTokenRequest(BaseModel):
    """Refresh token payload to obtain new access + refresh tokens."""

    refresh_token: str = Field(
        ..., description = "Refresh token from a previous login or refresh"
    )
    
class ChangePasswordRequest(BaseModel):
    """Change the current user's password, typically on first login."""

    current_password: str = Field(
        ..., min_length = 8, description = "Existing password for the authenticated user"
    )
    new_password: str = Field(
        ..., min_length = 8, description = "Replacement password (minimum 8 characters)"
    )


class Token(BaseModel):
    """Authentication response model for session credentials."""

    access_token: str = Field(
        ..., description = "Session access credential used for authenticated API requests"
    )
    refresh_token: str = Field(
        ...,
        description = "Session refresh credential used to renew an expired access credential",
    )
    token_type: str = Field(
        ..., description = "Credential type for the Authorization header, always 'bearer'"
    )
    must_change_password: bool = Field(
        ..., description = "True when the user must change the seeded default password"
    )
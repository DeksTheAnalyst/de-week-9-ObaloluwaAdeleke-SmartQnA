class LLMAPIError(Exception):
    """
    Custom exception for API-related errors.

    """
    def __init__(self, message: str, status_code: int = None):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)
    
    def __str__(self):
        if self.status_code:
            return f"LLMAPIError (Status {self.status_code}): {self.message}"
        return f"LLMAPIError: {self.message}"
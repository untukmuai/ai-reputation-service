import os
from typing import Optional

from web3 import Web3

from utils.libs_loader import libs_loader


class SomniaReferralService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return

        rpc_url = os.getenv("SOMNIA_RPC_URL")
        if not rpc_url:
            raise ValueError("Environment variable 'SOMNIA_RPC_URL' is required")

        raw_contract_address = os.getenv("SOMNIA_REF_CONTRACT_ADDRESS")
        if not raw_contract_address:
            raise ValueError("Environment variable 'SOMNIA_REF_CONTRACT_ADDRESS' is required")

        self.web3 = Web3(Web3.HTTPProvider(rpc_url))
        if not self.web3.is_connected():
            raise ConnectionError("Could not connect to Somnia RPC")

        try:
            contract_address = Web3.to_checksum_address(raw_contract_address)
        except ValueError as exc:
            raise ValueError("Invalid 'SOMNIA_REF_CONTRACT_ADDRESS'") from exc

        abi = libs_loader.get('ForUAIOpenMintWithReferralV4')
        self.contract = self.web3.eth.contract(
            address=contract_address,
            abi=abi,
        )

        self._initialized = True

    def normalize_address(self, address: str) -> Optional[str]:
        if not address:
            return None
        try:
            cleaned = address.strip()
            if len(cleaned) > 42:
                cleaned = cleaned[:42]
            return Web3.to_checksum_address(cleaned.lower())
        except Exception:
            return None

    def get_referral_count(self, address: str) -> int:
        normalized = self.normalize_address(address)
        if not normalized:
            return 0
        try:
            return self.contract.functions.referralCount(normalized).call()
        except Exception as exc:
            raise RuntimeError("Failed to fetch referral count") from exc
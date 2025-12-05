"""PLC I/O helper functions using python-snap7.

This module provides communication with Siemens S7 PLCs (S7-1200/1500).
The PLC should have a Data Block (DB) with all HAI tags as REAL values.

You must:
- Install python-snap7 (see requirements.txt)
- Adjust PLC_IP, RACK, SLOT in config.py
- Create a DB in TIA Portal matching the OFFSETS in config.py
- Set DB to "Optimized block access" = OFF for direct addressing

Reference: https://github.com/icsdataset/hai
"""

import struct
import time
from typing import Dict, List, Optional

import snap7
from snap7.util import get_real, set_real

from .config import PLC_IP, RACK, SLOT, DB_HAI_NUM, OFFSETS, DB_SIZE_BYTES


class PLCClient:
    """Wrapper around snap7 client for HAI tag read/write operations."""

    def __init__(self):
        self.client = snap7.client.Client()
        self._connected = False

    def connect(self) -> bool:
        """Connect to the PLC."""
        if not self._connected:
            try:
                self.client.connect(PLC_IP, RACK, SLOT)
                self._connected = self.client.get_connected()
            except Exception as e:
                print(f"[PLC] Connection error: {e}")
                self._connected = False
        return self._connected

    def disconnect(self):
        """Disconnect from the PLC."""
        if self._connected:
            try:
                self.client.disconnect()
            except Exception:
                pass
            self._connected = False

    def is_connected(self) -> bool:
        """Check if connected to PLC."""
        return self._connected and self.client.get_connected()

    # =========================================================================
    # Low-level read/write operations
    # =========================================================================

    def read_real(self, db_num: int, byte_offset: int) -> float:
        """Read a REAL (4 bytes) from a DB."""
        data = self.client.db_read(db_num, byte_offset, 4)
        return float(get_real(data, 0))

    def write_real(self, db_num: int, byte_offset: int, value: float) -> None:
        """Write a REAL (4 bytes) to a DB."""
        data = bytearray(4)
        set_real(data, 0, float(value))
        self.client.db_write(db_num, byte_offset, data)

    # =========================================================================
    # Tag-based operations (using OFFSETS from config)
    # =========================================================================

    def read_tag_real(self, tag_name: str) -> float:
        """Read a HAI tag value from the PLC."""
        if tag_name not in OFFSETS:
            raise KeyError(f"Unknown tag: {tag_name}")
        offset = OFFSETS[tag_name]
        return self.read_real(DB_HAI_NUM, offset)

    def write_tag_real(self, tag_name: str, value: float) -> None:
        """Write a HAI tag value to the PLC."""
        if tag_name not in OFFSETS:
            raise KeyError(f"Unknown tag: {tag_name}")
        offset = OFFSETS[tag_name]
        self.write_real(DB_HAI_NUM, offset, value)

    # =========================================================================
    # Batch operations (more efficient for many tags)
    # =========================================================================

    def read_all_tags(self) -> Dict[str, float]:
        """Read all HAI tags from the PLC in one operation."""
        # Read entire DB in one shot
        data = self.client.db_read(DB_HAI_NUM, 0, DB_SIZE_BYTES)
        
        values = {}
        for tag_name, offset in OFFSETS.items():
            if offset + 4 <= len(data):
                values[tag_name] = float(get_real(data, offset))
            else:
                values[tag_name] = 0.0
        return values

    def write_multiple_tags(self, values: Dict[str, float]) -> None:
        """Write multiple HAI tags to the PLC.
        
        Note: For best performance, this reads the entire DB, modifies
        the values, and writes back. For frequent updates, consider
        individual writes or optimize for your use case.
        """
        # Read current DB state
        data = bytearray(self.client.db_read(DB_HAI_NUM, 0, DB_SIZE_BYTES))
        
        # Modify values
        for tag_name, value in values.items():
            if tag_name in OFFSETS:
                offset = OFFSETS[tag_name]
                if offset + 4 <= len(data):
                    set_real(data, offset, float(value))
        
        # Write back
        self.client.db_write(DB_HAI_NUM, 0, data)

    def read_tags(self, tag_names: List[str]) -> Dict[str, float]:
        """Read specific HAI tags from the PLC."""
        values = {}
        for tag_name in tag_names:
            try:
                values[tag_name] = self.read_tag_real(tag_name)
            except Exception as e:
                print(f"[PLC] Error reading {tag_name}: {e}")
                values[tag_name] = 0.0
        return values


def ensure_connected(client: PLCClient, retry_interval: float = 2.0, max_retries: int = 0):
    """Keep trying to connect to PLC until successful.
    
    Args:
        client: PLCClient instance
        retry_interval: Seconds between retry attempts
        max_retries: Maximum retries (0 = infinite)
    """
    attempts = 0
    while True:
        try:
            if client.connect():
                print(f"[PLC] Connected to {PLC_IP}")
                return True
        except Exception as exc:
            print(f"[PLC] Connection failed: {exc}")
        
        attempts += 1
        if max_retries > 0 and attempts >= max_retries:
            print(f"[PLC] Max retries ({max_retries}) reached. Giving up.")
            return False
        
        print(f"[PLC] Retrying in {retry_interval}s...")
        time.sleep(retry_interval)

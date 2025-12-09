// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title SlideChain - On-chain provenance for MEDI-SLATE slides
contract SlideChain {
    struct SlideRecord {
        uint256 lectureId;     // 1..23
        uint256 slideId;       // 1..N per lecture
        string slideHash;      // keccak256 hash of SlideX.json as 0x...
        string uri;            // off-chain path (e.g. "Lecture 1/Slide1.json")
        uint256 timestamp;     // block timestamp when registered
        address registrant;    // who registered this slide
    }

    // key = keccak256(lectureId, slideId)
    mapping(bytes32 => SlideRecord) private records;

    event SlideRegistered(
        uint256 indexed lectureId,
        uint256 indexed slideId,
        string slideHash,
        string uri,
        address indexed registrant,
        uint256 timestamp
    );

    /// @notice Compute storage key used for a (lectureId, slideId) pair
    function _key(uint256 lectureId, uint256 slideId) internal pure returns (bytes32) {
        return keccak256(abi.encodePacked(lectureId, slideId));
    }

    /// @notice Register a slide hash for a (lectureId, slideId) pair
    /// @dev Reverts if already registered to keep provenance immutable.
    function registerSlide(
        uint256 lectureId,
        uint256 slideId,
        string calldata slideHash,
        string calldata uri
    ) external {
        require(lectureId > 0, "lectureId must be > 0");
        require(slideId > 0, "slideId must be > 0");

        bytes32 key = _key(lectureId, slideId);
        SlideRecord storage existing = records[key];
        require(existing.timestamp == 0, "slide already registered");

        records[key] = SlideRecord({
            lectureId: lectureId,
            slideId: slideId,
            slideHash: slideHash,
            uri: uri,
            timestamp: block.timestamp,
            registrant: msg.sender
        });

        emit SlideRegistered(
            lectureId,
            slideId,
            slideHash,
            uri,
            msg.sender,
            block.timestamp
        );
    }

    /// @notice Get a slide record for (lectureId, slideId)
    function getSlide(uint256 lectureId, uint256 slideId)
        external
        view
        returns (SlideRecord memory)
    {
        bytes32 key = _key(lectureId, slideId);
        return records[key];
    }

    /// @notice Check if a slide is registered
    function isRegistered(uint256 lectureId, uint256 slideId) external view returns (bool) {
        bytes32 key = _key(lectureId, slideId);
        return records[key].timestamp != 0;
    }
}

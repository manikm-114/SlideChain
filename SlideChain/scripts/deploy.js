const hre = require("hardhat");
const fs = require("fs");
const path = require("path");

async function main() {
  const SlideChain = await hre.ethers.getContractFactory("SlideChain");
  const slideChain = await SlideChain.deploy();
  await slideChain.waitForDeployment();

  const address = await slideChain.getAddress();
  console.log("SlideChain deployed to:", address);

  // Hardhat artifact path
  const artifactsPath = path.join(
    __dirname,
    "..",
    "artifacts",
    "contracts",
    "SlideChain.sol",
    "SlideChain.json"
  );
  const artifact = JSON.parse(fs.readFileSync(artifactsPath, "utf8"));

  // NEW OUTPUT → scripts/
  const outDir = path.join(__dirname); // __dirname is already "scripts"
  
  // Write ABI
  fs.writeFileSync(
    path.join(outDir, "SlideChain_abi.json"),
    JSON.stringify(artifact.abi, null, 2),
    "utf8"
  );

  // Write address
  fs.writeFileSync(
    path.join(outDir, "SlideChain_address.txt"),
    address,
    "utf8"
  );

  console.log("ABI and address written to scripts/SlideChain_abi.json and scripts/SlideChain_address.txt");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

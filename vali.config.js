module.exports = {
    "apps": [
      {
        "name": "bittensor-validator-subnet-410",
        "script": "neurons/validator.py",
        "interpreter": "python3",
        "args": [
          "--netuid", "410",
          "--wallet.name", "vali",
          "--wallet.hotkey", "default",
          "--blocks_per_epoch", "100",
          "--sample_min", "10",
          "--device", "cpu",
          "--logging.debug",
          "--subtensor.network", "test"
        ]
      }
   ]
}
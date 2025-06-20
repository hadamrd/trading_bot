import logging

# Create a logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create handlers (one for file and one for console)
file_handler = logging.FileHandler('test_models.log')
console_handler = logging.StreamHandler()

# Create formatters and add them to the handlers
formatter = logging.Formatter('%(asctime)s - %(message)s', '%m/%d/%Y %I:%M:%S')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)


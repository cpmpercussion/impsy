from impsy import interaction


def test_interaction_server():
    """Just tests creation of an interaction server object"""
    interaction_server = interaction.InteractionServer()


def test_logging():
    """Just sets up logging"""
    interaction.setup_logging(2)

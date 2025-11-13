import os
from robyn import Robyn
from controllers.dna_controller import DNAController
from controllers.identifi_controller import IdentifiController
from controllers.persona_controller import PersonaController
from controllers.tweet_controller import TweetController
from utils.libs_loader import libs_loader

app = Robyn(__file__)

print("Initializing AI Rep Service")
libs_loader.load_all()

DNAController(app)
PersonaController(app)
IdentifiController(app)
TweetController(app)

if __name__ == "__main__":
    app.start(host="0.0.0.0", port=8080)

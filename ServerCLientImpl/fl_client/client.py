# fl_client/client.py
import flwr as fl
from flwr.common import Scalar
import tensorflow as tf
import os
import logging
from rich.logging import RichHandler
from rich.console import Console
from typing import Dict

# --- Setup Logging ---
console = Console()
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, console=console)],
)
log = logging.getLogger("rich")

# --- Load Data ---
log.info("Loading MNIST dataset...")
# Access keras dynamically to avoid type analysis issues
keras = getattr(tf, "keras")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
log.info("Dataset loaded successfully.")

# --- Load Model from Shared Utils ---
# Ensure fl_server/utils.py is accessible or copied here.
# For Docker, this is handled by the build context.
try:
    # This path is relative to the app's root in the container
    from utils import create_model
    model = create_model()
    log.info("Model created successfully.")
except ImportError:
    log.error("Could not import `create_model` from `utils`. Make sure `utils.py` is in the same directory.")
    exit()


# --- Flower Client Implementation ---
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.client_id = client_id

    def get_parameters(self, config):
        log.info(f"[Client {self.client_id}] Sending model parameters to the server.")
        return model.get_weights()

    def fit(self, parameters, config):
        log.info(f"[Client {self.client_id}] Received parameters from server. Starting local training...")
        # Track download size from server (incoming parameters)
        try:
            down_bytes = sum(w.nbytes for w in parameters)
        except Exception:
            down_bytes = 0
        # Show a quick status while applying parameters
        with console.status("Applying server parameters...", spinner="line"):
            model.set_weights(parameters)
        # Use a subset of data for demonstration to simulate client data differences
        with console.status("Training locally (1 epoch)...", spinner="dots"):
            model.fit(x_train, y_train, epochs=1, batch_size=32, verbose=0, validation_split=0.1)
        log.info(f"[Client {self.client_id}] Local training finished.")
        new_weights = model.get_weights()
        # Track upload size to server (outgoing updated weights)
        try:
            up_bytes = sum(w.nbytes for w in new_weights)
        except Exception:
            up_bytes = 0
        metrics: Dict[str, Scalar] = {"up_bytes": int(up_bytes), "down_bytes": int(down_bytes)}
        return new_weights, len(x_train), metrics

    def evaluate(self, parameters, config):
        log.info(f"[Client {self.client_id}] Received parameters for evaluation.")
        # Track download size during evaluation, too
        try:
            down_bytes = sum(w.nbytes for w in parameters)
        except Exception:
            down_bytes = 0
        with console.status("Applying parameters for evaluation...", spinner="line"):
            model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        log.info(
            f"[Client {self.client_id}] Evaluation complete. "
            f"Accuracy: [bold cyan]{accuracy:.4f}[/bold cyan]"
        )
        return loss, len(x_test), {"accuracy": accuracy, "down_bytes": int(down_bytes)}


if __name__ == "__main__":
    server_host = os.environ.get("FLOWER_SERVER_HOST", "127.0.0.1")
    server_port = os.environ.get("FLOWER_SERVER_PORT", "8080")
    client_id = os.environ.get("CLIENT_ID", "0")
    server_address = f"{server_host}:{server_port}"

    log.info(f"Starting Flower client {client_id}...")
    log.info(f"Attempting to connect to server at [bold green]{server_address}[/bold green]")

    try:
        fl.client.start_numpy_client(
            server_address=server_address,
            client=FlowerClient(client_id=client_id),
        )
        log.info(f"Client {client_id} disconnected.")
    except Exception as e:
        log.error(f"[Client {client_id}] Connection failed: {e}", exc_info=True)
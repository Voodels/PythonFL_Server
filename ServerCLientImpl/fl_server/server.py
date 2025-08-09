# fl_server/server.py
import flwr as fl
import os
import logging
from rich.logging import RichHandler
from rich.panel import Panel
from rich.live import Live
from rich.table import Table
from rich.console import Console
from datetime import datetime

# Setup rich console and logging
console = Console()
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, console=console)],
)
log = logging.getLogger("rich")

# --- Rich UI Components ---
class ServerUI:
    def __init__(self):
        self.table = Table(title="Federated Learning Server Status")
        self.table.add_column("Time", justify="center", style="cyan")
        self.table.add_column("Event", justify="left", style="magenta")
        self.table.add_column("Details", justify="left", style="green")
        self.live = Live(self.table, screen=True, vertical_overflow="visible")
        self.current_tasks = {}

    def log(self, event: str, details: str = ""):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.table.add_row(timestamp, event, details)
        # Also emit to standard logs so it appears in docker logs
        try:
            log.info(f"{event} | {details}")
        except Exception:
            pass

    def __enter__(self):
        self.live.start()
        header = Panel(
            "[bold green]Flower Federated Learning Server[/bold green]",
            title="🚀 Launching",
            border_style="dim",
        )
        console.print(header)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Stop live view
        self.live.stop()
        if exc_type:
            log.error("Server shut down due to an error.", exc_info=True)
        else:
            log.info("✅ Server has shut down gracefully.")

    def spinner(self, text: str):
        """Return a simple status spinner context manager."""
        return console.status(text, spinner="dots")

# Custom Strategy to integrate with the UI
class RichStrategy(fl.server.strategy.FedAvg):
    def __init__(self, ui: ServerUI, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ui = ui

    def initialize_parameters(self, client_manager):
        self.ui.log("Strategy", "Initializing global model parameters...")
        return super().initialize_parameters(client_manager)

    def configure_fit(self, server_round, parameters, client_manager):
        self.ui.log(f"Round {server_round}", "Configuring `fit` tasks for clients...")
        with self.ui.spinner(f"Dispatching fit tasks (round {server_round})"):
            return super().configure_fit(server_round, parameters, client_manager)
    
    def aggregate_fit(self, server_round, results, failures):
        if not results:
            self.ui.log(f"Round {server_round}", "[red]No results received from clients.[/red]")
            return None, {}

        self.ui.log(
            f"Round {server_round}",
            f"Aggregating results from {len(results)} clients...",
        )
        # Show quick aggregation spinner
        agg_msg = f"Aggregating round {server_round} updates"
        # Compute bytes transferred metric from clients if provided
        total_up = 0
        total_down = 0
        try:
            for _, fitres in results:
                if hasattr(fitres, "metrics") and isinstance(fitres.metrics, dict):
                    # Prefer explicit keys if present
                    total_up += int(fitres.metrics.get("up_bytes", fitres.metrics.get("bytes", 0)))
                    total_down += int(fitres.metrics.get("down_bytes", 0))
        except Exception:
            total_up = 0
            total_down = 0

        with self.ui.spinner(agg_msg):
            aggregated = super().aggregate_fit(server_round, results, failures)
        if total_up > 0 or total_down > 0:
            up_mb = total_up / (1024 * 1024)
            down_mb = total_down / (1024 * 1024)
            details = []
            if total_up > 0:
                details.append(f"[blue]↑ updates[/blue] ~[bold]{up_mb:.2f} MB[/bold]")
            if total_down > 0:
                details.append(f"[green]↓ params[/green] ~[bold]{down_mb:.2f} MB[/bold]")
            self.ui.log(f"Round {server_round}", " | ".join(details))
        
        # Log any failures
        if failures:
            self.ui.log(
                f"Round {server_round}",
                f"[yellow]Warning: {len(failures)} clients failed.[/yellow]",
            )

        return aggregated

    def configure_evaluate(self, server_round, parameters, client_manager):
        self.ui.log(f"Round {server_round}", "Configuring `evaluate` tasks...")
        return super().configure_evaluate(server_round, parameters, client_manager)
        
    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            self.ui.log(f"Round {server_round}", "[red]No evaluation results.[/red]")
            return None, {}

        loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        # Handle case where metrics might be empty or missing accuracy key
        if metrics and "accuracy" in metrics:
            accuracy = metrics["accuracy"]
            self.ui.log(
                f"Round {server_round}",
                f"AGGREGATED ACCURACY: [bold cyan]{accuracy:.4f}[/bold cyan]",
            )
            return loss, {"accuracy": accuracy}
        else:
            self.ui.log(f"Round {server_round}", "[yellow]No accuracy metrics available.[/yellow]")
            return loss, metrics

if __name__ == "__main__":
    port = os.environ.get("FLOWER_PORT", "8080")
    server_address = f"0.0.0.0:{port}"
    num_rounds = 3
    min_clients = 2 # Number of clients to wait for

    with ServerUI() as ui:
        ui.log("Server Setup", f"Listening on {server_address}")
        ui.log("Server Setup", f"Waiting for {min_clients} clients to connect...")
        
        strategy = RichStrategy(
            ui=ui,
            min_fit_clients=min_clients,
            min_evaluate_clients=min_clients,
            min_available_clients=min_clients,
        )

        try:
            fl.server.start_server(
                server_address=server_address,
                config=fl.server.ServerConfig(num_rounds=num_rounds),
                strategy=strategy,
            )
        except Exception as e:
            ui.log("[bold red]FATAL ERROR[/bold red]", str(e))
import os
import sys
from typing import Optional, Tuple, List, Dict, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer, PreTrainedModel

from utils.file_utils import create_training_folder, save_losses, save_config
from utils.logging_utils import Logger, plot_losses, plot_predictions
from utils.time_utils import EpochTimer


class Trainer:
    def __init__(
        self,
        model: PreTrainedModel,
        optimiser: torch.optim.Optimizer,
        loss_fn: torch.nn.modules.loss._Loss,
        training_hyperparameters: Dict,
        tokenizer: PreTrainedTokenizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ):
        """Constructor class for Trainer used to train a transformer model for language modelling and text generation
        Args:
            model (nn.Module): Model to train
            optimiser (torch.optim.Optimizer): Optimiser to use for training
            loss_fn (torch.nn.modules.loss._Loss): Loss function to use for training
            training_hyperparameters (Dict): Dictionary containing training hyperparameters
            tokeniser (Dict): Dictionary containing encoder/decoder dictionaries and functions
            scheduler (Optional[torch.optim.lr_scheduler._LRScheduler], optional): Learning rate scheduler.
            Defaults to None.
        """

        self.model = model
        self.optimiser = optimiser
        self.loss_fn = loss_fn
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.best_model_dict = None

        # Preallocate variables defined in set_training_hyperparameters
        self.device = None
        self.epochs = None
        self.batch_size = None
        self.eval_every = None
        self.eval_iters = None
        self.max_seq_len = None
        self.save_every = None

        # Create a folder to save the model and training losses
        self.path = create_training_folder()

        # Unpack training hyperparameters
        self.set_training_hyperparameters(**training_hyperparameters)

        # Move the model to the device
        self.model.to(self.device)

        # Save the training hyperparameters as a  txt file
        save_config(training_hyperparameters, f"{self.path}/config.txt")

        # Save the model architecture as a txt file
        with open(f"{self.path}/model.txt", "w") as f:
            f.write(str(self.model))

        # Save the parameters of the model as a txt file
        save_config(count_parameters(self.model), f"{self.path}/model_parameters.txt")

        # Save the tokenizer
        self.tokenizer.save_pretrained(self.path)
        self.logger = None

    def train(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
        save_model: bool = True,
        save_model_path: Optional[str] = None,
        plotting: bool = True,
        verbose: bool = True,
        early_stopping: bool = False,
        early_stopping_patience: int = 10,
    ):
        """Train the model

        Args:
            train_dataloader (torch.utils.data.DataLoader): Training dataloader
            val_dataloader (torch.utils.data.DataLoader): Validation dataloader
            save_model (bool, optional): Whether to save the model. Defaults to True.
            save_model_path (Optional[str], optional): Path to save the model. Defaults to None.
            plotting (bool, optional): Whether to plot the training and validation losses. Defaults to True.
            verbose (Optional[bool], optional): Whether to print the training and validation losses. Defaults to True.
            early_stopping (bool, optional): Whether to use early stopping. Defaults to False.
            early_stopping_patience (int, optional): Patience for early stopping. Defaults to 10.
        """
        train_losses = []
        val_losses = []
        lowest_val_loss = float("inf")
        stop_training = False
        logger = Logger(
            self.path + "/training_logs/training_log.txt",
            name="training_log",
            verbose=verbose,
        )
        self.logger = logger

        logger.log_info(
            f"Training {type(self.model).__name__} for {self.epochs} iterations"
        )

        try:
            timer = EpochTimer()
            timer.start()

            count = 0

            for i in range(self.epochs):
                train_loss = self.training_loop(train_dataloader, method="train")
                train_losses.append(train_loss)

                val_loss = self.training_loop(val_dataloader, method="val")
                val_losses.append(val_loss)

                if i % self.eval_every == 0:
                    logger.log_info(
                        f"At Iteration: {i + 1}/{self.epochs}, Train loss: {train_loss: ,.4f}, "
                        f"Val loss: {val_loss: ,.4f}"
                    )

                    timer.lap()
                    logger.log_info(
                        timer.print_last_epoch_time(
                            label=f"Time taken for last {self.eval_every} iteration(s): "
                        )
                    )

                # Update the best model state dict and lowest validation loss
                lowest_val_loss, count = self.update_best_model_dict(
                    val_loss, lowest_val_loss, count
                )

                if early_stopping and i > 0 and count >= early_stopping_patience:
                    logger.log_info(f"Stopping early after {i + 1} iterations")
                    stop_training = True

                if self.save_every is not None and i % self.save_every == 0:
                    self.save_model(
                        f"{self.path}/saved_models/{type(self.model).__name__}_iter_{i + 1}/"
                    )

                if save_model and count == 0:
                    save_model_path = self.save_best_model(save_model_path)

                if stop_training:
                    break

            timer.lap()
            logger.log_info(timer.print_total_time(label="Total time taken: "))

            if save_model:
                # Save the losses
                save_losses(train_losses, val_losses, self.path)
                logger.log_info(
                    f"Saved losses at: {self.path}/training_logs/losses.csv"
                )

            self.model.load_state_dict(self.best_model_dict)
            logger.log_info(f"Saved best model at: {save_model_path}")

            if plotting:
                plot_save_path = (
                    f"{self.path}/training_logs/{type(self.model).__name__}_losses.png"
                    if save_model
                    else None
                )

                plot_losses(
                    train_losses,
                    val_losses,
                    model_name=type(self.model).__name__,
                    saved_path=plot_save_path,
                )
        except Exception as e:
            logger.log_error(f"Error while training: {str(e)}")
            raise e

        except KeyboardInterrupt:
            logger.log_info("Training interrupted by the user")
            # Exit the program
            sys.exit()

        return self.model, train_losses, val_losses

    def training_loop(
        self, dataloader: torch.utils.data.DataLoader, method: str = "train"
    ) -> float:
        """Training loop for the model

        Args:
            dataloader (torch.utils.data.DataLoader): DataLoader for the training data
            method (str, optional): Whether to train or validate. Defaults to 'train'.

        Returns:
            float: Average loss over the training loop
        """
        if method == "train":
            self.model.train()
        elif method == "val":
            self.model.eval()
        else:
            raise ValueError(f"method must be either 'train' or 'val' not {method}")

        total_loss = 0
        # Only want to loop through a subset of the data_loader as it is too large

        for batch_idx, batch in enumerate(dataloader):
            inputs = batch[0].to(self.device)
            attention_masks = batch[1].to(self.device)
            targets = batch[2].to(self.device)

            if method == "train":
                self.optimiser.zero_grad()

            outputs = self.model(inputs, attention_masks)[0].squeeze()

            # want to reshape the outputs and targets to be 2D with the same number of columns
            if self.model.config.num_labels == 1:
                # outputs = self.model(inputs).squeeze(-1)
                # targets = targets.squeeze(-1)
                loss = self.loss_fn(outputs, targets)
            else:
                loss = self.loss_fn(
                    outputs.view(-1, outputs.size(-1)), targets.view(-1)
                )

            # only want to backpropagate if we are training
            if method == "train":
                loss.backward()
                self.optimiser.step()

                # step the scheduler if it is not None
                if self.scheduler is not None:
                    self.scheduler.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def update_best_model_dict(
        self, loss_val: float, lowest_val_loss: float, count: int
    ) -> Tuple[float, int]:
        """Update the best model dictionary if the validation loss is the lowest so far
        Args:
            loss_val (float): Dictionary containing the training and validation losses
            lowest_val_loss (float): Lowest validation loss so far
            count (int): Number of times the validation loss has not been lower than the lowest validation loss.
            If this exceeds the early stopping patience, training will stop
        Returns:
            float: The updated lowest validation loss
        """
        if loss_val < lowest_val_loss:
            # Update the lowest validation loss
            lowest_val_loss = loss_val
            # Save the model state dict
            self.best_model_dict = self.model.state_dict()
            count = 0

        else:
            count += 1

        return lowest_val_loss, count

    def calculate_test_loss(
        self, test_data: DataLoader, log_error: bool = True
    ) -> float:
        """Calculate the loss on the full test data (without sampling)
        Args:
            test_data (Union[torch.Tensor, DataLoader]): Test data
            log_error (bool, optional): Whether to log the error. Defaults to True.
        Returns:
            float: Loss on the test data
        """
        test_loss = self.training_loop(test_data, method="val")
        if log_error:
            self.logger.log_info(f"Test loss: {test_loss}")

        return test_loss

    def log_numerical_outputs(
        self,
        dataloader: torch.utils.data.DataLoader,
        log_name: Optional[str] = None,
        output_type: str = "num",
    ):
        """Log the numerical outputs of the model to a file. It also plots the predictions vs the targets
        Args:
            dataloader (DataLoader): DataLoader for the data
            log_name (Optional[str], optional): Name of the log file. Defaults to None.
            output_type (str): Type of output. Either 'num' or 'text'

        """
        self.model.eval()
        if log_name is None:
            log_name = "float_predictions.txt"
        file_name = f"{self.path}/training_logs/{log_name}"
        with torch.no_grad():
            test_loss = 0
            predictions = []
            targets = []
            for batch_idx, (inputs, target) in enumerate(dataloader):
                inputs = inputs.to(self.device)

                if output_type == "num":
                    output = self.model(inputs).squeeze(-1)
                    target = target.to(self.device).squeeze(-1)
                    loss = self.loss_fn(output, target)
                else:
                    output = self.model(inputs)
                    target = target.to(self.device)
                    # print(output.shape, target.shape)
                    loss = self.loss_fn(
                        output.view(-1, output.size(-1)), target.view(-1)
                    )

                test_loss += loss.item()

                if output_type == "text":
                    # We want to append argmax of the output
                    predictions.extend(output.argmax(-1).tolist())
                    targets.extend(target.tolist())
                else:
                    predictions.extend(output.view(-1).tolist())
                    targets.extend(target.view(-1).tolist())
                if batch_idx % 5:
                    for i in range(len(inputs)):
                        with open(file_name, "a+") as f:
                            f.write(
                                "Question is "
                                + "".join(self.decode_data(inputs[i].tolist()))
                                + "\n"
                            )
                            if output_type == "text":
                                f.write(
                                    "Target is "
                                    + "".join(self.decode_data(target[i, :].tolist()))
                                    + "\n"
                                )
                                pred = "".join(
                                    self.decode_data(output[i, :].argmax(-1).tolist())
                                )
                                f.write(f"Prediction is {pred}" + "\n\n")
                            else:
                                f.write(
                                    "Target is "
                                    + str(round(target[i].item(), 4))
                                    + "\n"
                                )
                                pred = output[i].tolist()
                                f.write(f"Prediction is {pred:,.4f}" + "\n\n")

            test_loss /= len(dataloader)
            self.logger.log_info(f"Test loss was {test_loss :,.4f}")

            plot_save_path = (
                f"{self.path}/training_logs/{type(self.model).__name__}_predictions.png"
            )

            if output_type == "text":
                predictions, targets, count, error_log = self.convert_string_to_float(
                    predictions, targets
                )
                if count > 0:
                    self.logger.log_warning(
                        f"Could not convert {count} predictions to floats"
                    )
                    self.logger.log_warning(error_log)
                # log MSE error
                self.logger.log_info(
                    f"MSE Error on converted numerical outputs "
                    f"is {nn.MSELoss()(torch.tensor(predictions), torch.tensor(targets)) :,.4f}"
                )

            plot_predictions(
                predictions=predictions,
                targets=targets,
                model_name=type(self.model).__name__,
                saved_path=plot_save_path,
            )

    def convert_string_to_float(
        self, predictions: List[str], targets: List[str]
    ) -> Tuple[List[float], List[float], int, str]:
        """Convert the predictions and targets from strings to floats
        Args:
            predictions (List[str]): List of predicted tokens
            targets (List[str]): List of targets
        Returns:
            Tuple[List[float],List[float], int, str]: Tuple containing the converted predictions and targets, the number of errors and the error log
        """
        # convert each element in both lists to numbers. Decode and convert to float. If there is an error
        # Then remove that index from both lists and raise a warning
        pred_out = []
        target_out = []
        count = 0
        error_log = ""

        for i in range(len(predictions)):
            try:
                pred = "".join(self.decode_data(predictions[i]))
                pred = pred.split("<eos>")[0].replace("<sos>", "")
                pred_out.append(float(pred))
                target_out.append(float("".join(self.decode_data(targets[i]))))

            except ValueError:
                count += 1
                if count <= 20:
                    # Only want to log the first 20 errors otherwise the log file gets too big
                    error_log += f"Could not convert Prediction: {pred} to float.\n"
                    error_log += (
                        f"Target was {float(''.join(self.decode_data(targets[i])))}\n\n"
                    )
                continue
        return pred_out, target_out, count, error_log

    def set_training_hyperparameters(self, **kwargs):
        """Set the training hyperparameters which are passed as keyword arguments
        Args:
            **kwargs: Training hyperparameters
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

    def save_model(self, model_path: str):
        """Save the model
        Args:
            model_path (str): Path to save the model
        """
        # Ensure the directory exists
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        # Save the configuration
        config_path = os.path.join(model_path, "config.json")
        self.model.config.to_json_file(config_path)

        # Save the model weights
        self.model.save_pretrained(model_path)

    def save_best_model(self, best_model_path: Optional[str]):
        """Save the best model
        Args:
            best_model_path (Optional[str]): Path to save the best model
        """
        if best_model_path is None:
            best_model_path = (
                f"{self.path}/saved_models/{type(self.model).__name__}_best/"
            )
        self.save_model(best_model_path)
        return best_model_path

    def encode_data(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode the data using the tokenizer

        Args:
            texts (List[str]): List of texts to encode
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing the input ids and attention masks
        """
        input_ids = []
        attention_masks = []

        for text in texts:
            encode = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_seq_len,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids.append(encode["input_ids"][0])
            attention_masks.append(encode["attention_mask"][0])

        return torch.stack(input_ids), torch.stack(attention_masks)

    def decode_data(
        self, input_ids: torch.Tensor, skip_special_tokens: bool = True
    ) -> str:
        """Decode the input ids using the tokenizer
        Args:
            input_ids (torch.Tensor): Input ids to decode
            skip_special_tokens (bool, optional): Whether to skip the special tokens. Defaults to True.
        Returns:
            str: Decoded text
        """

        return self.tokenizer.decode(input_ids, skip_special_tokens=skip_special_tokens)


def set_seed(seed: Optional[int] = 0):
    """Set the random seed for reproducibility
    Args:
        seed (Optional[int], optional): Random seed. Defaults to 0.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if "numpy" in sys.modules:
        sys.modules["numpy"].random.seed(seed)

    if "random" in sys.modules:
        sys.modules["random"].seed(seed)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Counts the parameters of the model and returns a dictionary

    Args:
        model (nn.Module): Model to count the parameters of
    Returns:
        Dict[str, int]: Dictionary of the parameter counts
    """
    counts = {}
    for name, module in model.named_modules():
        count = sum(p.numel() for p in module.parameters() if p.requires_grad)
        counts[name] = count

    counts["total"] = counts.pop("")  # Move the main model's count to the "total" key
    return counts

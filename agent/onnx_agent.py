import os
import numpy as np
import onnxruntime as ort


class ONNXAgent:
    def __init__(self, model_path: str, num_threads: int = 1):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model file not found: {model_path}")

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = num_threads
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        self.session = ort.InferenceSession(
            model_path, opts, providers=["CPUExecutionProvider"]
        )

        self.inputs = self.session.get_inputs()
        self.outputs = self.session.get_outputs()

        self.input_names = [inp.name for inp in self.inputs]
        self.output_names = [out.name for out in self.outputs]

        self.obs_input_name = self.input_names[0]
        self.action_output_name = self.output_names[0]

        self.is_recurrent = len(self.input_names) > 1 and "lstm" in self.input_names[1].lower()

    def predict(self, observation: np.ndarray, state=None, episode_start=None, deterministic: bool = True):
        obs = np.asarray(observation, dtype=np.float32)
        is_single = (obs.ndim == 1)
        if is_single:
            obs = np.expand_dims(obs, axis=0)

        feed_dict = {self.obs_input_name: obs}

        if self.is_recurrent:
            if state is None:
                batch_size = obs.shape[0]
                h_shape = self.inputs[1].shape
                hidden_dim = h_shape[2] if len(h_shape) > 2 and isinstance(h_shape[2], int) else 64
                h_in = np.zeros((1, batch_size, hidden_dim), dtype=np.float32)
                c_in = np.zeros((1, batch_size, hidden_dim), dtype=np.float32)
                state = (h_in, c_in)

            feed_dict[self.input_names[1]] = state[0]
            feed_dict[self.input_names[2]] = state[1]

            results = self.session.run(self.output_names, feed_dict)
            action = results[0]
            next_state = (results[1], results[2])
        else:
            results = self.session.run([self.action_output_name], feed_dict)
            action = results[0]
            next_state = None

        if is_single and action.shape[0] == 1:
            action = action[0]

        return action, next_state

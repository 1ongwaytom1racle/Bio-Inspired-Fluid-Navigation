import numpy as np
import collections

class StateProcessor:
    """Processes environmental states and action history to generate inputs for the model.
    Note: Pressure data is not used in this version.
    """
    def __init__(self, reward_pos=(-4.0, -1), obstacle_pos=(-4.5, 0), pressure_dim=40, action_history_length=3, history_length=20, reward_scale_factor=10.0, max_distance=10.0):
        """Initialize the state processor.

        Args:
            reward_pos (tuple): Center coordinates of the reward target.
            obstacle_pos (tuple): Center coordinates of the obstacle.
            pressure_dim (int): Dimension of pressure data (not used in this version).
            action_history_length (int): Length of action history.
            history_length (int): Length of historical data (originally for pressure/visual, pressure unused now).
            reward_scale_factor (float): Reward scaling factor.
            max_distance (float): Maximum possible distance in the environment, used for normalization.
        """
        self.reward_pos = np.array(reward_pos)
        self.obstacle_pos = np.array(obstacle_pos)
        self.pressure_dim = pressure_dim
        self.action_history_length = action_history_length
        self.history_length = history_length
        self.sampled_pressure_history_length = history_length // 2  # Downsampled history length (pressure unused)
        self.reward_scale_factor = reward_scale_factor
        self.max_distance = max_distance

    def _get_fish_reference_frame(self, current_fish_data):
        """Calculate the fish's local coordinate system based on current sensor coordinates.
        
        x-axis: Direction of the line connecting the 1st and 5th data points
        Origin: 4th data point
        y-axis: Perpendicular to the x-axis
        
        Args:
            current_fish_data (list): List of current fish sensor coordinates, e.g., [[x1,y1], [x2,y2], ...].

        Returns:
            tuple: (Fish's center origin, Rotation matrix from world to fish local coordinates)
        """
        if len(current_fish_data) < 5:
            return np.array([0, 0]), np.eye(2)

        fish_points = np.array(current_fish_data)
        
        # Origin at the 4th data point (index 3)
        fish_origin = fish_points[3]

        # x-axis defined by the line between 1st and 5th data points (indices 0 and 4)
        p1 = fish_points[0]
        p5 = fish_points[4]
        
        x_axis_vec = p1 - p5
        norm = np.linalg.norm(x_axis_vec)
        if norm < 1e-6:
            x_axis = np.array([1, 0])  # Default x-axis
        else:
            x_axis = x_axis_vec / norm

        # y-axis perpendicular to x-axis (rotated 90 degrees counterclockwise)
        y_axis = np.array([-x_axis[1], x_axis[0]])

        # Rotation matrix from world to local coordinates
        inv_rotation_matrix = np.array([x_axis, y_axis])
        
        return fish_origin, inv_rotation_matrix

    def calculate_target_angle_distance(self, current_fish_data, current_reward_pos):
        """Calculate normalized distance to reward target and encode target angle as sin/cos components.

        Args:
            current_fish_data (list): Current fish sensor coordinates.
            current_reward_pos (np.ndarray): Current reward position.

        Returns:
            np.ndarray: Array of shape (3,) containing [cos(angle), sin(angle), normalized distance].
        """
        fish_origin, world_to_local_rotation = self._get_fish_reference_frame(current_fish_data)

        vec_to_reward = np.array(current_reward_pos) - fish_origin
        reward_local_pos = world_to_local_rotation @ vec_to_reward
        
        # Calculate angle (in radians)
        angle_rad = np.arctan2(reward_local_pos[1], reward_local_pos[0])
        
        # Calculate distance
        distance = np.linalg.norm(vec_to_reward)
        
        # Normalize distance
        normalized_distance = min(distance / self.max_distance, 1.0)

        return np.array([np.cos(angle_rad), np.sin(angle_rad), normalized_distance], dtype=np.float32)

    def build_state_from_history(self, env_id, state_recorder, done, current_reward_pos):
        """Retrieve data from recorder and build a complete state.
        
        Args:
            env_id (str): Environment ID.
            state_recorder (MultiEnvStateRecorder): Instance of state recorder.
            done (bool): Whether current state is terminal.
            current_reward_pos (np.ndarray): Current reward position of the environment.
        
        Returns:
            tuple or None: (Pressure history [unused], Visual info, Action/velocity features) or None (insufficient data).
        """
        with state_recorder.env_lock:
            state_history = state_recorder.env_states.get(env_id)
            action_history = state_recorder.env_actions.get(env_id)

            if not state_history or len(state_history) < 1:
                return None
            
            if done and len(state_history) < 20:
                return None

        return self.process(state_history, action_history, current_reward_pos)

    def process(self, state_history: collections.deque, action_history: collections.deque, current_reward_pos):
        """Integrate historical data to generate a complete state for the model.
        Pad with zeros if history is insufficient to maintain fixed shape.

        Args:
            state_history (deque): Deque containing past state records.
            action_history (deque): Deque containing past action records.
            current_reward_pos (np.ndarray): Current reward position of the environment.

        Returns:
            tuple: (Pressure history [unused], Latest visual info, Action and velocity features)
                   Shapes: (sampled_pressure_history_length, pressure_dim), (3,), ((action_history_length - 1) + 3,)
        """
        # 1. Initialize zero matrices
        pressure_history = np.zeros((self.sampled_pressure_history_length, self.pressure_dim), dtype=np.float32)
        latest_visual_info = np.zeros(3, dtype=np.float32)
        action_history_out = np.zeros((self.action_history_length - 1, 1), dtype=np.float32)

        # 2. Populate pressure history (unused) and extract latest visual info
        if state_history:
            full_recent_states = list(state_history)
            sampled_states = full_recent_states[::-2][::-1]

            num_to_take = len(sampled_states)
            
            # Extract and populate pressure data (unused)
            try:
                extracted_pressures = []
                for s in sampled_states:
                    pressure_data = s['fish_data'][0]['pressures']
                    
                    if len(pressure_data) >= 8:
                        selected_points = [pressure_data[1], pressure_data[4], pressure_data[7]]
                        remaining_points = pressure_data[8:]
                        pressure_data = selected_points + remaining_points

                    if len(pressure_data) > self.pressure_dim:
                        pressure_data = pressure_data[:self.pressure_dim]
                    elif len(pressure_data) < self.pressure_dim:
                        pressure_data = pressure_data + [0.0] * (self.pressure_dim - len(pressure_data))
                    extracted_pressures.append(pressure_data)
                
                extracted_pressures = np.array(extracted_pressures, dtype=np.float32)
                pressure_history[-num_to_take:] = extracted_pressures
            except (IndexError, KeyError):
                pass

            # Calculate angle-distance data for the latest state
            try:
                latest_state = state_history[-1]
                coords = latest_state['fish_data'][0]['coordinates']
                latest_visual_info = self.calculate_target_angle_distance(coords, current_reward_pos)
            except (IndexError, KeyError):
                pass

        # 3. Populate action history
        if len(action_history) >= 2:
            num_to_take_actions = min(len(action_history), self.action_history_length)
            recent_actions = list(action_history)[-num_to_take_actions:]

            if len(recent_actions) >= 2:
                try:
                    amplitudes = [a['amplitude'] for a in recent_actions]
                    diffs = np.diff(amplitudes).astype(np.float32)
                    action_history_out[-len(diffs):] = np.expand_dims(diffs, axis=1)
                except KeyError:
                    pass

        # 4. Calculate velocity and angular momentum
        final_velocities = np.zeros(3, dtype=np.float32)
        if action_history and state_history:
            last_action_time = action_history[-1]['time']
            relevant_states = [s for s in state_history if s['time'] >= last_action_time]

            if relevant_states:
                # Calculate average linear velocity
                sum_vx, sum_vy = 0.0, 0.0
                count = 0
                for state in relevant_states:
                    try:
                        vel = state['velocities'][0]
                        sum_vx += vel['x']
                        sum_vy += vel['y']
                        count += 1
                    except (IndexError, KeyError):
                        continue

                # Extract latest angular momentum
                latest_wz = 0.0
                try:
                    latest_wz = state_history[-1]['rotational_momentums'][0]['z'] * 1000.0
                except (IndexError, KeyError):
                    pass

                # Calculate velocity projections
                if count > 0:
                    avg_vx = sum_vx / count
                    avg_vy = sum_vy / count
                    
                    try:
                        latest_coords = state_history[-1]['fish_data'][0]['coordinates']
                        fish_origin, _ = self._get_fish_reference_frame(latest_coords)
                        
                        reward_direction_vector = current_reward_pos - fish_origin
                        norm = np.linalg.norm(reward_direction_vector)
                        
                        if norm > 1e-6:
                            v_parallel_unit = reward_direction_vector / norm
                            v_perpendicular_unit = np.array([-v_parallel_unit[1], v_parallel_unit[0]])
                            avg_v_raw = np.array([avg_vx, avg_vy])
                            
                            proj_parallel = np.dot(avg_v_raw, v_parallel_unit)
                            proj_perpendicular = np.dot(avg_v_raw, v_perpendicular_unit)
                            
                            final_velocities[0] = proj_parallel
                            final_velocities[1] = proj_perpendicular
                    except (IndexError, KeyError):
                        pass
                
                final_velocities[2] = latest_wz

        # 5. Concatenate into final feature vector
        action_history_flat = action_history_out.flatten()
        final_feature_vector = np.concatenate([action_history_flat, final_velocities])

        return pressure_history, latest_visual_info, final_feature_vector

    def is_data_sufficient(self, state_history: collections.deque, action_history: collections.deque):
        """Check if historical data is sufficient for decision-making"""
        state_count = len(state_history) if state_history else 0
        action_count = len(action_history) if action_history else 0
        
        return (state_count >= self.history_length and 
                action_count >= self.action_history_length)

    def calculate_reward(self, state_history, action_history, current_reward_pos):
        """Calculate reward based on average metrics within a window.
        Reward = (Average velocity projection) - (Average power consumption) + Angle guidance reward
        
        Args:
            state_history (deque): Historical state records.
            action_history (deque): Historical action records.
            current_reward_pos (np.ndarray): Current reward position of the episode.
        """
        if len(action_history) < 2 or len(state_history) < 2:
            return 0.0

        try:
            action_list = list(action_history)
            start_time = action_list[-2]['time']
            
            relevant_states = []
            for state in reversed(state_history):
                if state['time'] >= start_time:
                    relevant_states.append(state)
                else:
                    break
            
            if len(relevant_states) < 2:
                return 0.0
            
            relevant_states.reverse()

            # 1. Calculate average velocity vector within the window
            velocities = [np.array([s['velocities'][0]['x'], s['velocities'][0]['y']]) for s in relevant_states]
            avg_velocity = np.mean(velocities, axis=0)

            # 2. Calculate vector from fish to reward center
            latest_fish_coords = relevant_states[-1]['fish_data'][0]['coordinates']
            fish_origin, _ = self._get_fish_reference_frame(latest_fish_coords)
            vec_to_reward = np.array(current_reward_pos) - fish_origin
            distance_to_reward = np.linalg.norm(vec_to_reward)

            # 3. Calculate velocity projection in reward direction
            if distance_to_reward < 1e-6:
                velocity_projection = 0.0
            else:
                velocity_projection = np.dot(avg_velocity, vec_to_reward) / distance_to_reward
            
            # 4. Calculate average power consumption within the window
            total_powers = []
            for s in relevant_states:
                power_data = s['power_data'][0]
                constraint_power = np.linalg.norm([power_data['constraint']['x'], power_data['constraint']['y']])
                inertia_power = np.linalg.norm([power_data['inertia']['x'], power_data['inertia']['y']])
                total_powers.append(constraint_power + inertia_power)
            
            avg_power = np.mean(total_powers)

            
            # 6. Calculate velocity and energy based reward
            velocity_reward = (velocity_projection - avg_power) 
            
            # 7. Final reward
            reward = float(velocity_reward)

            return reward * self.reward_scale_factor

        except (KeyError, IndexError) as e:
            print(f"[Reward] State or action data format error: {e}")
            return 0.0
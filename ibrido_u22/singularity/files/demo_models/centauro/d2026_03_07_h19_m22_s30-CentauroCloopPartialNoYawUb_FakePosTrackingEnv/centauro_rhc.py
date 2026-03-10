from aug_mpc.controllers.rhc.horizon_based.hybrid_quad_rhc import HybridQuadRhc

import numpy as np

from typing import Dict 

from centaurohybridmpc.utils.sysutils import PathsGetter

class CentauroRhc(HybridQuadRhc):

    def __init__(self, 
            srdf_path: str,
            urdf_path: str,
            robot_name: str, # used for shared memory namespaces
            codegen_dir: str,
            n_nodes: float = 31,
            dt: float = 0.03,
            injection_node: int = 10,
            max_solver_iter = 1, # defaults to rt-iteration
            open_loop: bool = True,
            close_loop_all: bool = False,
            dtype = np.float32, 
            verbose = False, 
            debug = False,
            refs_in_hor_frame = True,
            timeout_ms: int = 60000,
            custom_opts: Dict = {}
            ):

        paths = PathsGetter()
        self._files_suffix=""
        if open_loop:
            self._files_suffix="_open"
        
        self._add_upper_body=False
        if ("add_upper_body" in custom_opts) and \
            (custom_opts["add_upper_body"]):
            self._add_upper_body=True
            self._files_suffix+="_ub"

        config_path=paths.RHCCONFIGPATH_NO_WHEELS+self._files_suffix+".yaml"
        
        super().__init__(srdf_path=srdf_path,
            urdf_path=urdf_path,
            config_path=config_path,
            robot_name=robot_name, # used for shared memory namespaces
            codegen_dir=codegen_dir, 
            n_nodes=n_nodes,
            dt=dt,
            injection_node=injection_node,
            max_solver_iter=max_solver_iter, # defaults to rt-iteration
            open_loop=open_loop,
            close_loop_all=close_loop_all,
            dtype=dtype,
            verbose=verbose, 
            debug=debug,
            refs_in_hor_frame=refs_in_hor_frame,
            timeout_ms=timeout_ms,
            custom_opts=custom_opts)
        
        self._fail_idx_scale=1e-9
        self._fail_idx_thresh_open_loop=1e0
        self._fail_idx_thresh_close_loop=10
        
        if open_loop:
            self._fail_idx_thresh=self._fail_idx_thresh_open_loop
        else:
            self._fail_idx_thresh=self._fail_idx_thresh_close_loop

        # adding some additional config files for jnt imp control
        self._rhc_fpaths.append(paths.JNT_IMP_CONFIG_XBOT+".yaml")
        self._rhc_fpaths.append(paths.JNT_IMP_CONFIG+".yaml")
        
    def _set_rhc_pred_idx(self):
        self._pred_node_idx=round((self._n_nodes-1)*2/3)

    def _set_rhc_cmds_idx(self):
        self._rhc_cmds_node_idx=2

    def _config_override(self):
        paths = PathsGetter()
        if ("control_wheels" in self._custom_opts):
            if self._custom_opts["control_wheels"]:
                self.config_path = paths.RHCCONFIGPATH_WHEELS+self._files_suffix+".yaml"
                if ("fix_yaw" in self._custom_opts) and \
                    (self._custom_opts["fix_yaw"]):
                    self.config_path = paths.RHCCONFIGPATH_WHEELS_NO_YAW+self._files_suffix+".yaml"
                if ("replace_continuous_joints" in self._custom_opts) and \
                    (not self._custom_opts["replace_continuous_joints"]):
                    # use continuous joints -> different config
                    self.config_path = paths.RHCCONFIGPATH_WHEELS_CONTINUOUS+self._files_suffix+".yaml"
                    if ("fix_yaw" in self._custom_opts) and \
                        (self._custom_opts["fix_yaw"]):
                        self.config_path = paths.RHCCONFIGPATH_WHEELS_CONTINUOUS_NO_YAW+self._files_suffix+".yaml"

        else:
            self._custom_opts["control_wheels"]=False

        if not self._custom_opts["control_wheels"]:
            self._fixed_jnt_patterns=self._fixed_jnt_patterns+\
                ["j_wheel", 
                "ankle_yaw"]
            self._custom_opts["replace_continuous_joints"]=True
        
    def _init_problem(self):
        
        if not self._custom_opts["control_wheels"]:
            self._yaw_vertical_weight=120.0
        else:
            self._yaw_vertical_weight=50.0

        fixed_jnts_patterns=[
            "d435_head",
            "velodyne_joint",
            "dagana"]
        
        if not self._add_upper_body:
            fixed_jnts_patterns.append("j_arm")
            fixed_jnts_patterns.append("torso")

        if ("fix_yaw" in self._custom_opts) and \
            (self._custom_opts["fix_yaw"]):
            fixed_jnts_patterns.append("ankle_yaw")

        flight_duration_sec=0.5 # [s]
        flight_duration=int(flight_duration_sec/self._dt)
        post_flight_duration_sec=0.2 # [s]
        post_flight_duration=int(post_flight_duration_sec/self._dt)

        step_height=0.1
        if ("step_height" in self._custom_opts):
            step_height=self._custom_opts["step_height"]

        super()._init_problem(fixed_jnt_patterns=fixed_jnts_patterns,
            wheels_patterns=["wheel_"],
            foot_linkname="wheel_1",
            flight_duration=flight_duration,
            post_flight_stance=post_flight_duration,
            step_height=step_height,
            keep_yaw_vert=True,
            yaw_vertical_weight=self._yaw_vertical_weight,
            vertical_landing=True,
            vertical_land_weight=10.0,
            phase_force_reg=5e-2,
            vel_bounds_weight=1.0)
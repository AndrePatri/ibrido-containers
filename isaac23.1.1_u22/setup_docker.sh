# Byobu Fix for launching BASH instead of SH
mkdir -p /root/.byobu/
echo 'set -g default-shell /bin/bash' >>/root/.byobu/.tmux.conf
echo 'set -g default-command /bin/bash' >>/root/.byobu/.tmux.conf

show git branches from terminal
echo 'parse_git_branch() {' >> /root/.bashrc && \
echo '    git branch 2> /dev/null | sed -e "/^[^*]/d" -e "s/* \(.*\)/ (\1)/"' >> /root/.bashrc && \
echo '}' >> /root/.bashrc && \
echo 'export PS1="\u@\h \[\033[32m\]\w\[\033[33m\]$(parse_git_branch)\[\033[00m\] $ "' >> /root/.bashrc

# isaac setup (just create symbolic link for setting up pyenv)
# mkdir -p ~/.local/share/ov/pkg/isaac_sim-2023.1.1/
# ln -s /isaac-sim/setup_python_env.sh ~/.local/share/ov/pkg/isaac_sim-2023.1.1/setup_python_env.sh
# ln -s /isaac-sim/setup_conda_env.sh ~/.local/share/ov/pkg/isaac_sim-2023.1.1/setup_conda_env.sh
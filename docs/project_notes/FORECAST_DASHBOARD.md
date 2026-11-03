# 5+12 预测看板

看板只读服务器 `outputs/next_5plus12/` 的逐期预测及独立评分文件。浏览、切换期号和切换曲线都不会训练模型或更新开奖数据。没有记录时显示空状态。开奖后运行预测器的 `--evaluate 期号` 再刷新页面即可查看命中个数与 Brier 评分。

服务器独立环境：在仓库目录用 Python 3.11 创建虚拟环境，安装 `requirements-dashboard.txt`。启动：

    .venv/bin/python -m streamlit run forecast_dashboard.py --server.address 127.0.0.1 --server.port 18501 --server.headless true --browser.gatherUsageStats false --client.showSidebarNavigation false

本机访问使用 SSH 本地端口转发（也可使用 MobaXterm 的 Tunneling 面板）：

    ssh -N -L 127.0.0.1:18501:127.0.0.1:18501 -p 22118 ryang@10.70.5.64

浏览器打开 http://127.0.0.1:18501/ 。无需开放服务器公网端口。服务器使用 tmux 会话 `lottery-forecast-ui` 管理该应用。旧工作台另可从侧栏进入“预测看板”页面。

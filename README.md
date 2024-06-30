# HFL-DRL

For running HFL-DRL with GQUIC-256, model CNN on 1 server, 2 edges and 4 clients:

1. Install dependencies

- pip install -r requirements.txt

2. Start training

- Open 7 terminals
- First terminal: python server.py
- Second terminal: python head.py --edge_port 40001
- Third terminal: python head.py --edge_port 40002
- Terminal number 4,5,6,7: python distributed_client.py

##### You can config the training by modifying the parameters in options.py

##### For example:

- num_communication = 10
- num_edge_aggregation = 10
- num_local_update = 6

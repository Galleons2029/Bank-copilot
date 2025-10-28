#!/bin/sh
# rabbitmq-init.sh

# 等待 RabbitMQ 启动
sleep 10

# 创建虚拟主机
rabbitmqctl add_vhost /my_vhost

# 创建用户
rabbitmqctl add_user guest guest

# 给用户权限
rabbitmqctl set_permissions -p /my_vhost guest ".*" ".*" ".*"

# 创建队列
rabbitmqctl add_queue test_files --vhost=/my_vhost

# 创建交换机并绑定队列
rabbitmqctl add_exchange my_exchange direct --vhost=/my_vhost
rabbitmqctl bind_queue test_files my_exchange --vhost=/my_vhost
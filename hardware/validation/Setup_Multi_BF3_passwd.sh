#!/bin/bash

# Get the list of rshim devices
devices=$(ls /dev/ | grep -i 'rsh')

# Iterate over each device and run the expect script
for device in $devices; do
    /usr/bin/expect <<EOF
# Your expect script here
set user_input "$device"
set device "/dev/\$user_input/console"
set baud_rate "115200"
set username "ubuntu"
set password "ubuntu"
set root_access "sudo su -"
set root_passwd "passwd"
set set_ubuntu_passwd "$Passwd"
set set_root_passwd "$Root_Passwd"
set command "ip a | grep -i 'oob_net'"

# Start screen
puts "Starting screen session on \$device at baud rate \$baud_rate..."
spawn screen \$device -b \$baud_rate

# Expect the username prompt
expect "Username:"
send "\$username\r"

# Expect the password prompt
expect "Password:"
send "\$password\r"

expect "Current Password:"
send "\$password\r"




# Send the root access command
expect "#"
send "\$root_access\r"

# Expect the root password prompt
expect "Password:"
send "\$root_passwd\r"

# Expect the # prompt
expect "#"
send "\$set_root_passwd\r"

# Expect the # prompt
expect "#"
send "\$command\r"

# Print the output and exit
expect eof
EOF
done

for i in $(screen -ls | grep -i $(hostname) | awk '{print $1}'); do screen -X -S $i  quit; done


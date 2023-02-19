#!/bin/bash

# Check that the correct number of arguments has been passed
if [ $# -ne 2 ]; then
  echo "Usage: $0 <parent directory> <group>"
  exit 1
fi

user=$(whoami)
parent=$1
group=$2

# Check that <parent directory> exists
if [ ! -d "$parent" ]; then
  echo "Directory $parent does not exist"
  exit 1
fi

# Check that <group> is a valid group
if ! grep -q "^$group:" /etc/group; then
  echo "Group $group does not exist"
  exit 1
fi

# Check that the user is in <group>
if ! groups | grep -q "\b$group\b"; then
  echo "User $user is not in group $group"
  exit 1
fi

# Set group owner and permissions for the top-level directory
chown $user:$group $parent

# Set default permissions for parent and its subdirectories and files
find $parent -exec \
  setfacl -d -m \
  d:u::rwx,d:g:$group:rwx,d:o::r-x,\
  f:u::rw-,f:g:$group:rw-,f:o::r-- \
  {} +

# Allow execute permissions for the scripts/ directory, if it exists
if [ -d "$parent/scripts" ]; then
  find $parent/scripts -exec \
  setfacl -d -m \
  d:u::rwx,d:g:$group:rwx,d:o::r-x,\
  f:u::rwx,f:g:$group:rwx,f:o::r-x \
  {} +
fi

echo "Permissions set for $parent"
echo "Runtime: $SECONDS seconds"

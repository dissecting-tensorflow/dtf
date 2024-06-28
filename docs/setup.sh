# Bazel Cache
apt install nginx-extras -y

location /cache/ {
  # The path to the directory where nginx should store the cache contents.
  root /data00/son.nguyen/.cache/remote;
  # Allow PUT
  dav_methods PUT;
  # Allow nginx to create the /ac and /cas subdirectories.
  create_full_put_path on;
  # The maximum size of a single file.
  client_max_body_size 1G;
  allow all;
}
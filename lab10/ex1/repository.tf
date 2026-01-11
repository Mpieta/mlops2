locals {
  visibility = var.publicly_visible ? "public" : "private"
}

resource "github_repository" "example" {
  name        = var.repository_name 
  description = var.repository_description 
  visibility  = local.visibility
  auto_init   = true 
}
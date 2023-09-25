locals {
  identifier = lookup(local.target_identifiers, "dev", "default-val")
  target_identifiers = {
    dev = "dev123"
    stg = "stg456"
    prd = "prd789"
  }
}

variable "region" {
  type        = string
  description = "Region"
}

variable "prefix" {
  type        = string
  description = "Prefix applied to every resource name"
}

variable "is_production" {
  description = "Is production"
  type        = bool
  default     = false
}

variable "openai_api_key" {
  description = "OpenAI API Key, used by the agent and the judges"
  type        = string
}

variable "logfire_token" {
  description = "Logfire write token, used by the agent runner to send its spans"
  type        = string
  sensitive   = true
}

variable "logfire_read_token" {
  description = "Logfire read token, used by the request handler to fetch the spans back (logfire read-tokens create)"
  type        = string
  sensitive   = true
}

variable "vpc_id" {
  description = "VPC ID for Lambda deployment"
  type        = string
}

variable "private_subnet_ids" {
  description = "List of private subnet IDs for Lambda deployment"
  type        = list(string)
  sensitive   = true
}

variable "request_handler_security_group_id" {
  description = "Request handler security group ID for Lambda deployment"
  type        = string
  default     = null
}

variable "agent_runner_security_group_id" {
  description = "Agent runner security group ID for Lambda deployment"
  type        = string
  default     = null
}

variable "response_handler_security_group_id" {
  description = "Response handler security group ID for Lambda deployment"
  type        = string
  default     = null
}

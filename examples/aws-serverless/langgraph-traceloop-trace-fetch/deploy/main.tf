# One Lambda (no queue mode): it runs the agent on /chat, with Traceloop tracing, and reads the spans back on
# GET /fetch
module "serverless_agents" {
  source = "yaalalabs/ak-serverless/aws"
  version = "0.9.5"

  providers            = { aws = aws, docker = docker }
  prefix               = var.prefix
  region               = var.region
  is_production        = var.is_production
  vpc_id               = var.vpc_id
  private_subnet_ids   = var.private_subnet_ids
  product_display_name = "AK LangGraph Traceloop Trace Fetch Serverless Example"

  # GET /api/v1/fetch returns a session's spans from Traceloop; /chat is created as usual
  gateway_endpoints = [
    { path = "fetch", method = "GET" },
  ]

  request_handler = {
    function_name        = "traceloop-fetch"
    function_description = "Agent Kernel LangGraph Traceloop trace fetch example: /chat and /fetch"
    handler_path         = "lambda.handler"
    package_path         = "../dist"
    package_type         = "Image"
    memory_size          = 1024
    timeout              = 60
    security_group_id    = var.request_handler_security_group_id
    environment_variables = {
      OPENAI_API_KEY    = var.openai_api_key # the agent's model
      TRACELOOP_API_KEY = var.traceloop_api_key
      # Traceloop's own endpoint unless set; the SDK and the fetch both read it
      TRACELOOP_BASE_URL = var.traceloop_base_url
    }
  }
}

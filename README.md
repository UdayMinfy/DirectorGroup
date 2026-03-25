# Lambda Research Assistant

This project is an AI-powered research API built on AWS Lambda.

In simple terms, a user sends a question, the system looks for useful information on the web, reads the source pages, prepares a concise answer, and returns the result as a JSON API response. At the same time, it tracks token usage in PostgreSQL and enforces daily and monthly limits for each user.

This makes the project useful when you want an AI assistant that is not just answering questions, but also:

- uses website-based source material
- keeps a record of what was used
- tracks cost and usage in tokens
- controls how much each user can consume
- works like a backend service that other apps can call

## Who This Is For

This README is written so both technical and non-technical readers can understand the project.

If you are non-technical, the easiest way to think about this system is:

`User asks a question -> system researches the web -> AI writes an answer -> usage is saved -> quota is checked`

If you are technical, this repository contains:

- an AWS Lambda function
- a research agent flow
- website search and extraction
- AI summarization through Amazon Bedrock
- PostgreSQL-based quota and usage tracking

## What Problem This Project Solves

Many AI systems give answers, but they do not always:

- show where the information came from
- use fresh web content
- control how much each user is allowed to use
- keep detailed usage logs for reporting or billing

This project addresses that by combining research, summarization, and quota tracking into one backend workflow.

## What the System Does

When a request comes in, the project can:

1. receive a user question
2. verify the user
3. check daily and monthly quota
4. search the web or use direct URLs
5. open and extract content from websites
6. summarize the findings with an AI model
7. return the answer in JSON format
8. save request and token usage details in PostgreSQL

## Example Use Case

A frontend application, chatbot, or internal tool can send a prompt like:

`Give the top typing test websites and a brief description of each.`

The project will then:

- search relevant pages
- read those pages
- summarize the information
- return the answer
- store token usage for that user

## High-Level Architecture

At a high level, the project looks like this:

```text
Client / App
    |
    v
AWS Lambda API
    |
    v
Research Agent
    |----> Web Search
    |----> Browser Extraction
    |----> AI Summarization
    |
    v
PostgreSQL
    |----> users
    |----> plans
    |----> request logs
    |----> daily/monthly summaries
```

## Request Journey

Here is the full flow in easy language:

### 1. A request reaches the Lambda function

The main entry file is [app.py](/d:/Minfy%20Projects/DirectedGroup/lambdacode/src/app.py).

The request should contain:

- `prompt`: the user's question
- `user_id`: the user making the request

It may also contain:

- `urls`: optional list of websites to use directly instead of searching the web

If `prompt` or `user_id` is missing, the API returns a `400` error.

### 2. The system checks whether the user exists

User and quota logic is handled in [quota_service.py](/d:/Minfy%20Projects/DirectedGroup/lambdacode/src/quota_service.py).

If the user already exists, the system continues with that record.

If the user does not exist, the system:

- creates a user record
- assigns the user to the default token plan

### 3. A request log entry is created

Before the research starts, the system creates an initial request record in the database.

At this stage, the request is marked as `in_progress`.

This is useful because even if the request later fails or is blocked, there is still a record showing that it was attempted.

### 4. Quota is checked

The system checks how many tokens the user has already consumed:

- today
- this month

If the user has no remaining quota, the request is stopped and the API returns a `429` error.

### 5. The research agent starts work

The research flow is mainly handled in [agent.py](/d:/Minfy%20Projects/DirectedGroup/lambdacode/src/agent.py).

The agent has two ways to gather source material:

- use the URLs provided directly in the request
- search the web automatically if no URLs are given

### 6. Web search is performed if needed

Search logic is handled in [web_search.py](/d:/Minfy%20Projects/DirectedGroup/lambdacode/src/web_search.py).

The current setup uses Tavily as the search provider.

This step returns a list of relevant websites.

### 7. Website content is extracted

For each website, the system creates a task and tries to extract useful page content.

Each task gets:

- a task ID
- a title
- the page URL
- a status such as `completed` or `failed`

This makes it easier to understand which websites were successfully processed.

### 8. AI summarization happens

Once content is collected from the websites, the system sends the extracted material to the AI model through Amazon Bedrock.

The AI then creates the final answer based on the gathered sources.

### 9. Usage is saved

When the request is successful, the system stores:

- token usage
- answer preview
- model used
- request status
- daily and monthly summary totals

### 10. Final response is returned

The API returns a JSON response containing:

- the original prompt
- the final answer
- model ID
- token usage
- source websites
- task details
- request ID
- updated quota summary

## Example Request

```json
{
  "prompt": "Give the top typing test websites and a brief description of each.",
  "user_id": "demo-user-1"
}
```

## Example Successful Response

```json
{
  "statusCode": 200,
  "headers": {
    "Content-Type": "application/json"
  },
  "body": "{\"prompt\":\"Give the top typing test websites and a brief description of each.\",\"answer\":\"Here are the top typing test websites...\",\"model_id\":\"amazon.nova-pro-v1:0\",\"usage\":{\"input_tokens\":2733,\"output_tokens\":445,\"total_tokens\":3178}}"
}
```

## Example Quota Error Response

```json
{
  "statusCode": 429,
  "headers": {
    "Content-Type": "application/json"
  },
  "body": "{\"error\":\"Daily token quota exceeded.\",\"quota\":{\"daily_limit\":20000,\"monthly_limit\":300000,\"daily_used\":20000,\"monthly_used\":25000,\"daily_remaining\":0,\"monthly_remaining\":275000}}"
}
```

## What the API Response Means

A normal successful response usually includes these parts:

- `prompt`: the original question
- `answer`: the generated answer
- `model_id`: the AI model used
- `usage`: input, output, and total tokens
- `tasks`: details of website extraction attempts
- `sources`: preview of extracted source content
- `request_id`: a unique tracking ID for the request
- `quota`: the user's remaining token balance

## Project Structure

```text
lambdacode/
|-- src/
|   |-- app.py
|   |-- agent.py
|   |-- quota_service.py
|   |-- web_search.py
|   |-- tools.py
|   |-- config.py
|   |-- task_models.py
|   |-- browser_extractor.py
|   |-- bedrock_client.py
|   |-- agentcore_browser_client.py
|-- db/
|   |-- schema.sql
|   |-- seed_test_users.sql
|   |-- migrate_to_users2.sql
|   |-- inspect_quota.sql
|-- events/
|-- template.yaml
|-- requirements.txt
```

## Important Files Explained

[template.yaml](/d:/Minfy%20Projects/DirectedGroup/lambdacode/template.yaml)
This is the AWS SAM template. It defines the Lambda function, runtime, memory, timeout, permissions, and environment variables.

[app.py](/d:/Minfy%20Projects/DirectedGroup/lambdacode/src/app.py)
This is the main Lambda entry point. It receives the request, validates it, calls the quota service, runs the research agent, and returns the final API response.

[agent.py](/d:/Minfy%20Projects/DirectedGroup/lambdacode/src/agent.py)
This controls the research workflow: search or URL input, extraction, summarization, and response formatting.

[quota_service.py](/d:/Minfy%20Projects/DirectedGroup/lambdacode/src/quota_service.py)
This handles users, plans, request logging, quota checks, success/failure updates, and token summary updates.

[web_search.py](/d:/Minfy%20Projects/DirectedGroup/lambdacode/src/web_search.py)
This connects to the web search provider and returns search results.

[config.py](/d:/Minfy%20Projects/DirectedGroup/lambdacode/src/config.py)
This loads project settings from environment variables.

[schema.sql](/d:/Minfy%20Projects/DirectedGroup/lambdacode/db/schema.sql)
This creates the main PostgreSQL tables used by the project.

[seed_test_users.sql](/d:/Minfy%20Projects/DirectedGroup/lambdacode/db/seed_test_users.sql)
This inserts sample plans and users for testing.

## PostgreSQL Tables Explained Simply

This project uses 4 main tables. These are the tables we created for user and token tracking.

### 1. `token_plans`

Purpose:
Stores the rules for each usage plan.

Simple explanation:
This table defines how many tokens a plan allows per day and per month.

Why it matters:
Different users can have different limits depending on the plan they are assigned.

Example plans:

- `default`
- `tiny-daily`

### 2. `users2`

Purpose:
Stores user records.

Simple explanation:
This table keeps each user and links that user to one plan.

Why it matters:
Without this table, the system would not know who is making requests or which quota rules apply to them.

### 3. `user_token_usage_events`

Purpose:
Stores one row per request.

Simple explanation:
Every time a user sends a question, the system creates a record here.

It can store:

- the prompt
- answer preview
- token counts
- model used
- request status
- error message if something failed
- time of the request

Why it matters:
This is the detailed history table.

### 4. `user_token_usage_summary`

Purpose:
Stores usage totals by period.

Simple explanation:
Instead of counting every request again and again, this table stores daily and monthly totals for each user.

It can store:

- total input tokens used
- total output tokens used
- total tokens used
- total request count
- last request time

Why it matters:
This makes quota checks much faster and easier.

## In One Line: Table Purpose

- `token_plans` = plan limits
- `users2` = users and their assigned plans
- `user_token_usage_events` = detailed request history
- `user_token_usage_summary` = daily and monthly totals

## Technologies Used

- AWS Lambda
- AWS SAM
- Python 3.12
- Amazon Bedrock
- Tavily Search API
- PostgreSQL
- Psycopg

## Environment Variables

Important configuration values include:

- `BEDROCK_REGION`
- `BEDROCK_MODEL_ID`
- `BROWSER_REGION`
- `BROWSER_IDENTIFIER`
- `SEARCH_API_PROVIDER`
- `TAVILY_API_KEY`
- `SEARCH_RESULT_LIMIT`
- `MAX_PAGE_CHARS`
- `MAX_SOURCE_CHARS`
- `REQUEST_TIMEOUT_SECONDS`
- `DB_HOST`
- `DB_PORT`
- `DB_NAME`
- `DB_USER`
- `DB_PASSWORD`
- `DEFAULT_DAILY_TOKEN_LIMIT`
- `DEFAULT_MONTHLY_TOKEN_LIMIT`

These are mainly defined in [template.yaml](/d:/Minfy%20Projects/DirectedGroup/lambdacode/template.yaml) and read through [config.py](/d:/Minfy%20Projects/DirectedGroup/lambdacode/src/config.py).

## Running the Project Locally

### Prerequisites

Before running the project, make sure you have:

- Python installed
- AWS SAM CLI installed
- PostgreSQL installed and running
- AWS credentials configured
- access to Amazon Bedrock
- a Tavily API key

### Step 1: Install dependencies

```powershell
pip install -r requirements.txt
pip install -r .\src\requirements.txt
```

### Step 2: Set up the PostgreSQL database

Run the SQL in [schema.sql](/d:/Minfy%20Projects/DirectedGroup/lambdacode/db/schema.sql) to create the tables.

If you want sample users and plans for testing, run [seed_test_users.sql](/d:/Minfy%20Projects/DirectedGroup/lambdacode/db/seed_test_users.sql).

### Step 3: Configure environment values

Make sure the database and API values are correct for your machine or environment.

Pay special attention to:

- database host and port
- database username and password
- Tavily API key
- Bedrock region and model ID

### Step 4: Build the project

```powershell
sam build
```

### Step 5: Run locally

```powershell
sam local start-api
```

### Step 6: Test the API

```powershell
curl -X POST http://127.0.0.1:3000/ ^
  -H "Content-Type: application/json" ^
  -d "{\"prompt\":\"Give the top typing test websites and a brief description of each.\",\"user_id\":\"demo-user-1\"}"
```

## Quota Logic Explained Simply

The system measures AI usage in tokens.

Every request uses some tokens:

- input tokens for the text sent to the model
- output tokens for the text generated by the model
- total tokens as the combined count

The project checks:

- daily usage
- monthly usage

If the user is still under the limit:

- the request is allowed
- the answer is generated
- usage is saved

If the user has crossed the limit:

- the request is blocked
- the system marks it as blocked in the event log
- the API returns a `429` response

## Status Tracking

Each request can end in different states.

Common statuses include:

- `in_progress`
- `completed`
- `blocked`
- `failed`

This makes it easier to debug issues and understand what happened during a request.

## Common Error Cases

The API can return these common errors:

- `400` when `prompt` is missing
- `400` when `user_id` is missing
- `429` when token quota is exceeded
- `500` when search, extraction, summarization, or database work fails

## Why This Design Is Useful

This design is useful because it separates responsibilities clearly:

- Lambda handles the API request
- the research agent handles content gathering and summarization
- PostgreSQL stores users and usage
- quota logic protects the system from overuse

For a business or product team, this means:

- better control over usage
- easier tracking of cost
- better transparency on what sources were used
- easier future expansion into paid plans or user tiers

## Possible Future Improvements

This project already works as a research API, but it can be extended further with:

- user authentication
- admin dashboard for quota monitoring
- better source ranking
- response caching
- billing integration
- richer analytics
- support for more plans and user roles
- UI frontend for non-technical users

## Summary

This project is a controlled AI research backend.

It accepts a user question, gathers information from websites, summarizes it using AI, returns the answer as an API response, and saves token usage in PostgreSQL with daily and monthly quota enforcement.

For a non-technical reader, the simplest description is:

This is a backend service that works like a research assistant, but with rules, tracking, and usage limits built in.


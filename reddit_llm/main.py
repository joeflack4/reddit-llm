import praw
import anthropic
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import tiktoken

# Load environment variables
load_dotenv()


def setup_reddit_client():
    """Initialize Reddit client using environment variables."""
    return praw.Reddit(
        client_id=os.getenv('REDDIT_CLIENT_ID'),
        client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
        user_agent="PostFetcher/1.0"
    )


def count_tokens(text):
    """Count the number of tokens in the text using tiktoken."""
    enc = tiktoken.get_encoding("cl100k_base")  # Claude's encoding
    return len(enc.encode(text))


def fetch_subreddit_posts(subreddit_name, time_period_days=30, max_tokens=180000):
    """
    Fetch posts from a subreddit for the specified time period, respecting token limits.

    Args:
        subreddit_name (str): Name of the subreddit without 'r/'
        time_period_days (int): Number of days to look back
        max_tokens (int): Maximum number of tokens to collect

    Returns:
        list: List of post contents, each under token limit
    """
    reddit = setup_reddit_client()
    subreddit = reddit.subreddit(subreddit_name)
    cutoff_time = datetime.utcnow() - timedelta(days=time_period_days)

    posts_content = []
    current_chunk = []
    current_chunk_tokens = 0

    for post in subreddit.new(limit=None):
        post_time = datetime.fromtimestamp(post.created_utc)
        if post_time < cutoff_time:
            break

        post_text = f"""
Title: {post.title}
Time: {post_time.strftime('%Y-%m-%d %H:%M:%S UTC')}
URL: https://reddit.com{post.permalink}
Content:
{post.selftext}
---
"""
        post_tokens = count_tokens(post_text)

        # If adding this post would exceed the token limit, start a new chunk
        if current_chunk_tokens + post_tokens > max_tokens:
            if current_chunk:
                posts_content.append("\n".join(current_chunk))
            current_chunk = [post_text]
            current_chunk_tokens = post_tokens
        else:
            current_chunk.append(post_text)
            current_chunk_tokens += post_tokens

    # Add the last chunk if it exists
    if current_chunk:
        posts_content.append("\n".join(current_chunk))

    return posts_content


class RedditClaudeChat:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        self.conversation = None

    def start_conversation(self, context):
        """Start a new conversation with the given context."""
        system_prompt = """You are analyzing Reddit posts from a specific subreddit. 
        Answer questions about these posts based on the context provided. 
        If asked about something not in the context, clearly state that it's not in the provided posts."""

        self.conversation = self.client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=4096,
            temperature=0,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": f"Here are the Reddit posts to analyze:\n\n{context}"
                }
            ]
        )

        # Get acknowledgment from Claude
        return self.conversation.content

    def ask_question(self, question):
        """Ask a question in the existing conversation."""
        if not self.conversation:
            raise Exception("Conversation not started. Call start_conversation first.")

        self.conversation = self.client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=4096,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": question
                }
            ],
            conversation_id=self.conversation.conversation_id
        )

        return self.conversation.content


def main():
    # Get user inputs
    subreddit_name = input("Enter subreddit name (without r/): ")
    time_period = int(input("Enter number of days to look back (default 30): ") or 30)

    # Fetch posts
    print(f"Fetching posts from r/{subreddit_name}...")
    posts_chunks = fetch_subreddit_posts(subreddit_name, time_period)

    if not posts_chunks:
        print("No posts found in the specified time period.")
        return

    print(f"\nFetched {len(posts_chunks)} chunks of posts.")

    # Initialize chat for each chunk
    chats = []
    for i, chunk in enumerate(posts_chunks):
        print(f"\nInitializing conversation for chunk {i + 1}/{len(posts_chunks)}...")
        chat = RedditClaudeChat()
        response = chat.start_conversation(chunk)
        chats.append(chat)
        print(f"Chunk {i + 1} initialized successfully.")

    print("\nReady to answer questions! (Type 'quit' to exit)")

    while True:
        question = input("\nEnter your question about the posts: ")

        if question.lower() == 'quit':
            break

        # Query all chunks and combine responses
        print("\nAsking Claude...")
        for i, chat in enumerate(chats):
            print(f"\nResponse from chunk {i + 1}/{len(chats)}:")
            response = chat.ask_question(question)
            print(response)
            print("-" * 50)


if __name__ == "__main__":
    main()
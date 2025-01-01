"""Reddit LLM"""
from typing import List

import praw
import anthropic
from datetime import datetime, timedelta
import os

from anthropic.types import TextBlock
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

# todo: maybe increase this 30 days i did was like 155tokens estimate
def fetch_subreddit_posts(subreddit_name, time_period_days=30, max_tokens=180000):
    """
    Fetch posts from a subreddit for the specified time period, up to the token limit.
    
    Args:
        subreddit_name (str): Name of the subreddit without 'r/'
        time_period_days (int): Number of days to look back
        max_tokens (int): Maximum number of tokens to collect
    
    Returns:
        str: Combined post contents within token limit
        int: Number of posts included
        int: Number of posts skipped due to token limit
    """
    reddit = setup_reddit_client()
    subreddit = reddit.subreddit(subreddit_name)
    cutoff_time = datetime.utcnow() - timedelta(days=time_period_days)
    
    all_posts = []
    total_tokens = 0
    posts_included = 0
    posts_skipped = 0
    
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
        
        # Check if adding this post would exceed the token limit
        if total_tokens + post_tokens <= max_tokens:
            all_posts.append(post_text)
            total_tokens += post_tokens
            posts_included += 1
        else:
            posts_skipped += 1
    
    # Combine all posts into a single string
    combined_posts = "\n".join(all_posts)
    
    return combined_posts, posts_included, posts_skipped

class RedditClaudeChat:
    """Chat"""
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        self.conversation = None
        
    def question_with_context(self, context: str, subreddit_name: str, question: str):
        """Start a new conversation with the given context."""
        system_prompt = f"""You are analyzing posts from the r/{subreddit_name} subreddit.
        Answer questions about these posts based on the context provided.
        If asked about something not in the context, clearly state that it's not in the provided posts.
        When appropriate, include post titles and URLs in your responses to support your analysis."""
        
        self.conversation = self.client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=4096,
            temperature=0,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": f"Here are the Reddit posts to analyze from r/{subreddit_name}:\n\n{context}"
                },
                {
                    "role": "user",
                    "content": question
                }
            ]
        )
        
        return self.conversation.content

    # todo: can't persist chat so need to combine question with original method that makes convo
    # def ask_question(self, question: str, convo_id: str):
    #     """Ask a question in the existing conversation."""
    #     if not self.conversation and not convo_id:
    #         raise Exception("Conversation not started. Call start_conversation first.")
    #     if self.conversation and convo_id:
    #         raise Exception("Both conversation and convo_id provided. Use one or the other.")
    #     convo_id = convo_id or self.conversation.conversation_id
    #
    #     self.conversation = self.client.messages.create(
    #         model="claude-3-sonnet-20240229",
    #         max_tokens=4096,
    #         temperature=0,
    #         messages=[
    #             {
    #                 "role": "user",
    #                 "content": question
    #             }
    #         ],
    #         # TODO: bugfix AttributeError: 'Message' object has no attribute 'conversation_id'
    #         # conversation_id=self.conversation.conversation_id
    #         # TODO: TypeError: Messages.create() got an unexpected keyword argument 'id'
    #         id=convo_id
    #     )
    #
    #     return self.conversation.content

# TODO: parameterize this properly. last saved locally 2025/01/01
#  - Characters: 572446 ; est Tokens: 157790
def main(use_cache=True, save_cache=True):
    """Main"""
    # TODO: want to use pre-known convo with Claude but looks impossible...
    #  https://docs.anthropic.com/en/api/messages-examples Multiple conversational turns The Messages API is stateless,
    #  which means that you always send the full conversational history to the API.
    # convo_id = 'msg_019FuadLbbQYPNDFSsMhLYGg'
    subreddit_name = "PathOfExile2"
    convo_id = None
    chat = None

    if not convo_id:
        # Get user inputs
        subreddit_name = "PathOfExile2"
        time_period = 30  # todo: re-enable user input later
        # subreddit_name = input("Enter subreddit name (without r/) (default: PathOfExile2: ") or "PathOfExile2"
        # time_period = int(input("Enter number of days to look back (default 30): ") or 30)

        # todo: cache basd on subreddit
        if use_cache:
            with open("posts_cache.txt", "r") as f:
                posts_content = f.read()
            with open("posts_included.txt", "r") as f:
                posts_included = int(f.read())
            with open("posts_skipped.txt", "r") as f:
                posts_skipped = int(f.read())
        else:
            # Fetch posts
            print(f"\nFetching posts from r/{subreddit_name}...")
            posts_content, posts_included, posts_skipped = fetch_subreddit_posts(subreddit_name, time_period)

            if save_cache:
                with open("posts_cache.txt", "w") as f:
                    f.write(posts_content)
                with open("posts_included.txt", "w") as f:
                    f.write(str(posts_included))
                with open("posts_skipped.txt", "w") as f:
                    f.write(str(posts_skipped))

        if not posts_content:
            print("No posts found in the specified time period.")
            return

        print(f"\nFetched {posts_included} posts successfully.")
        if posts_skipped > 0:
            print(f"Skipped {posts_skipped} posts due to token limit.")

        # Initialize chat
        # TODO: dryify this part
        print("\nInitializing conversation with Claude...")
        chat = RedditClaudeChat()
        # noinspection PyUnusedLocal  temp_prolly_dont_need_var
        # response = chat.start_conversation(posts_content, subreddit_name)
        print("Conversation initialized successfully.")

        print("\nReady to answer questions! (Type 'quit' to exit)")
        print("Example questions:")
        print("- What are the main topics being discussed in this subreddit?")
        print("- What are the most controversial or debated topics?")
        print("- What features or changes are users requesting the most?")

    if not chat:
        print("\nInitializing conversation with Claude...")
        chat = RedditClaudeChat()
        print("\nReady to answer questions! (Type 'quit' to exit)")
        print("Example questions:")
        print("- What are the main topics being discussed in this subreddit?")
        print("- What are the most controversial or debated topics?")
        print("- What features or changes are users requesting the most?")
    while True:
        question = input("\nEnter your question about the posts: ")
        
        if question.lower() == 'quit':
            break
            
        print("\nAsking Claude...")
        # response = chat.ask_question(question, convo_id)
        response: List[TextBlock] = chat.question_with_context(posts_content, subreddit_name, question)
        print("\nClaude's response:")
        for block in response:
            print(block.text)


if __name__ == "__main__":
    main()

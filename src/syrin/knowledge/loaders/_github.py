"""GitHub loader for fetching repository information."""

from __future__ import annotations

import httpx

from syrin.knowledge._document import Document


class GitHubLoader:
    """Load information from GitHub repositories.

    Fetches repository metadata, README content, and optionally code files.
    Supports one repo, multiple repos, or all public repos for a user.

    Example:
        loader = GitHubLoader("octocat", repos=["Hello-World"])
        docs = await loader.aload()

        # Multiple repos
        loader = GitHubLoader("org", repos=["repo1", "repo2"], include_readme=True)

        # All public repos (repos=None)
        loader = GitHubLoader("username", repos=None)
    """

    def __init__(
        self,
        username: str,
        repos: list[str] | None = None,
        include_readme: bool = True,
        include_code: bool = False,
        token: str | None = None,
    ) -> None:
        """Initialize GitHubLoader.

        Args:
            username: GitHub username or organization.
            repos: List of repository names to fetch. None = all public repos for user.
            include_readme: Include README content (default: True).
            include_code: Include code files (default: False, can be large).
            token: Optional GitHub API token for private repos or higher rate limits.
        """
        self._username = username
        self._repos = repos
        self._include_readme = include_readme
        self._include_code = include_code
        self._token = token

    @property
    def username(self) -> str:
        """GitHub username or organization."""
        return self._username

    @property
    def repos(self) -> list[str] | None:
        """Repository names to fetch; None means all public repos."""
        return self._repos

    async def aload(self) -> list[Document]:
        """Load GitHub content asynchronously.

        Returns:
            List of Documents with repository information.
        """
        docs: list[Document] = []

        if self._repos is not None and len(self._repos) == 0:
            return docs
        if self._repos:
            for repo_name in self._repos:
                repo_docs = await self._fetch_repo(repo_name)
                docs.extend(repo_docs)
        else:
            user_docs = await self._fetch_user_repos()
            docs.extend(user_docs)

        return docs

    async def _fetch_repo(self, repo_name: str) -> list[Document]:
        """Fetch a single repository's information."""
        headers = self._get_headers()
        repo_url = f"https://api.github.com/repos/{self._username}/{repo_name}"

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(repo_url, headers=headers)
            response.raise_for_status()
            repo_data = response.json()

        content_parts = [
            f"# {repo_data['name']}",
            f"Description: {repo_data.get('description', 'No description')}",
            f"Language: {repo_data.get('language', 'Unknown')}",
            f"Stars: {repo_data.get('stargazers_count', 0)}",
            f"Forks: {repo_data.get('forks_count', 0)}",
            f"URL: {repo_data.get('html_url', '')}",
        ]

        if self._include_readme:
            readme_content = await self._fetch_readme(repo_name)
            if readme_content:
                content_parts.append("\n## README\n")
                content_parts.append(readme_content)

        return [
            Document(
                content="\n".join(content_parts),
                source=f"github/{self._username}/{repo_name}",
                source_type="github",
                metadata={"repo": repo_name, "username": self._username},
            )
        ]

    async def _fetch_readme(self, repo_name: str) -> str | None:
        """Fetch README for a repository."""
        headers = self._get_headers()
        readme_url = f"https://api.github.com/repos/{self._username}/{repo_name}/readme"

        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.get(readme_url, headers=headers)
                if response.status_code == 200:
                    import base64

                    data = response.json()
                    content = data.get("content", "")
                    if content:
                        return base64.b64decode(content).decode("utf-8")
            except Exception:
                pass
        return None

    async def _fetch_user_repos(self) -> list[Document]:
        """Fetch list of user's repositories (paginated, up to 500)."""
        headers = self._get_headers()
        repos: list[dict[str, object]] = []
        page = 1
        max_repos = 500

        async with httpx.AsyncClient(timeout=30.0) as client:
            while len(repos) < max_repos:
                url = (
                    f"https://api.github.com/users/{self._username}/repos?per_page=100&page={page}"
                )
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                batch = response.json()
                if not batch:
                    break
                repos.extend(batch)
                if len(batch) < 100:
                    break
                page += 1

        content_parts = [f"# Repositories for {self._username}\n"]

        for repo in repos:
            content_parts.append(
                f"## {repo['name']}\n"
                f"Description: {repo.get('description', 'No description')}\n"
                f"Language: {repo.get('language', 'Unknown')}\n"
                f"Stars: {repo.get('stargazers_count', 0)}\n"
                f"URL: {repo.get('html_url', '')}\n"
            )

        return [
            Document(
                content="\n".join(content_parts),
                source=f"github/{self._username}",
                source_type="github",
                metadata={"username": self._username},
            )
        ]

    def _get_headers(self) -> dict[str, str]:
        """Get HTTP headers with optional auth."""
        headers: dict[str, str] = {
            "Accept": "application/vnd.github.v3+json",
        }
        if self._token and self._token.strip():
            headers["Authorization"] = f"token {self._token}"
        return headers


__all__ = ["GitHubLoader"]

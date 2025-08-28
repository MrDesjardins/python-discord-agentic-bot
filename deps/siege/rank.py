import discord

siege_ranks = [
    "Champion",
    "Diamond",
    "Emerald",
    "Platinum",
    "Gold",
    "Silver",
    "Bronze",
    "Copper",
]

def get_user_rank_siege(user: discord.Member) -> str:
    """
    Check the user's roles to determine their rank
    """
    if user is None:
        return "Copper"

    for role in user.roles:
        if role.name in siege_ranks:
            return role.name
    return "Copper"
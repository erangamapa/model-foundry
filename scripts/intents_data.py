"""Hand-authored synthetic customer messages, 30 per intent, written for the first training run."""
import json
import random
from pathlib import Path

DATA = {
'place_bet': """Put $20 on Arsenal to win tonight
I want to bet 50 on the Lakers moneyline
Can you place a fiver on Djokovic for the final?
Back Liverpool at 2.10 for 10 euros
Lay 100 on over 2.5 goals in the Madrid game
Place a parlay: Chiefs win and Bills cover
Bet $15 on Verstappen to win the race
Get me 30 on the Yankees tonight
I'd like to wager 25 on Man City to beat Brighton
stake 40 on Celtics -4.5
Throw ten bucks on the underdog in the UFC main event
Please put 5 pounds on England to win the match
Take my bet: 200 on Federer at evens
Add a bet slip for both teams to score, 12 dollars
Whack 75 on the Warriors to win
Can I get a 20 dollar bet on the home team?
Make it a 60 euro bet on Bayern to win 3-0
Wager 8 on Nadal to take the first set
Submit a single of 100 on Dodgers money line
Go for 35 on under 220 points in the Nets game
I wanna put some money on the Rangers game
Bet the max on Tyson Fury by knockout
place 1 dollar on Tottenham to draw
Confirm my bet of 45 on Inter to win
Do a 3 fold accumulator, 10 stake, Spurs Villa Chelsea
Lock in 500 on Real Madrid to qualify
Back the Raptors with 22 bucks
Could you stake 18 on the over in the Packers game
Place 90 on Alcaraz in straight sets
Bet 14 on a draw in the derby""",
'check_balance': """How much money do I have in my account?
What's my balance?
Show me my current balance please
how much can I withdraw right now
Do I have enough funds to bet 50?
Check my wallet
What is my available balance after open bets?
tell me how much I've got left
Balance please
How much credit is left on my account
Can you see how much cash I have?
what's in my account right now
Is my deposit showing in my balance yet?
How many funds are available to me?
I need to know my account balance
Whats my bankroll at the moment
Did my 100 deposit arrive? What's the balance
how much have I got to play with
Show my wallet total
What is my remaining balance this week?
Please tell me my current funds
How much money is in my betting account
Check how much I have available to withdraw
What does my account say I have?
Am I in the red? What is my balance
balance check
How much is left in my account after yesterday?
Let me see my available cash
Do I have anything left in my wallet?
Tell me my funds and my pending amount""",
'odds_question': """What are the odds on Arsenal to win tonight?
What price is Djokovic for the final?
What's the spread for the Lakers game?
How much would I win if I bet 10 on Liverpool?
Are the odds on the Chiefs better than yesterday?
What's the over under for the Yankees game?
What are the chances of Man City winning the league?
Who is the favourite for the Grand Prix?
What odds do you have on a draw in the derby?
Give me the moneyline for the Celtics game
What do the odds say about Fury by knockout?
Is Real Madrid odds on or odds against?
What's the payout on a 20 dollar bet at 3.5?
how do decimal odds work
Which team is the underdog tonight?
What are the prices for both teams to score?
Show me the odds for Nadal vs Alcaraz
Has the line moved on the Packers game?
What would 5 dollars on Spurs at 4 to 1 return?
Is it better to bet the over or under at these odds?
What odds are offered on the Dodgers first inning?
Tell me the current price on Bayern
how likely is England to win according to the odds
what's the best price available for Inter
What are the handicap odds for Brighton?
How are the odds calculated?
what is the line on the UFC main event
Are there any boosted odds today?
What is the implied probability of 2.50 odds?
What's the market for first goalscorer tonight?""",
'bet_status': """Did my bet on Arsenal win?
What happened to my Lakers bet?
Has my bet been settled yet?
Check the status of my last bet
Is my parlay still alive?
Why hasn't my bet been paid out?
Where is the winnings from my Chelsea bet
Is my wager on Djokovic confirmed?
Show me my open bets
Did my accumulator come in?
my bet from yesterday still says pending
Was my bet accepted?
Can you tell me if the Yankees bet lost?
Have I been paid for the Liverpool win?
What's the result of my bet slip number 4482?
Is my cash out available on my current bet?
Did I win anything from last night's games?
My bet shows as void, what happened?
Status of my 50 dollar bet on Spurs please
Which of my bets are still running?
Has the Packers bet been graded?
Is my bet still active after the match was postponed?
Look up my bet history from this weekend
Did my Fury bet cash?
Why was my bet cancelled?
how are my open parlays doing
Tell me if my wager has settled
was I paid out for the Celtics game
Update me on my bet from last night
Is my bet on the derby a winner?""",
'set_limit': """I want to set a daily deposit limit of 100
Can you cap my weekly losses at 200?
Please limit my bets to 50 per day
Set a monthly spending limit of 500
I need to put a limit on how much I can lose
Lower my deposit limit to 25 a week
I'd like to take a break for 30 days
Can you block me from betting for a week
Reduce my maximum stake to 10
set a session time limit of two hours
I want to self exclude
Limit my deposits to 300 a month please
Add a loss limit of 150 per week
Please restrict my account so I can only bet 20 at a time
Cooling off period of 7 days please
I think I'm betting too much, set a limit
Change my daily limit from 200 to 100
Put a cap on my monthly losses of 1000
Turn on a reality check every 30 minutes
Increase my weekly deposit limit to 400
How do I set a spending cap on my account?
Set my max bet size to 5 dollars
I want to stop myself from depositing more than 50 a day
Please lock my account for the weekend
Make my deposit limit 75 per week
Can I limit my wagering to 100 a month
Impose a stake limit of 15 on all bets
Enable a loss limit for me
I would like to reduce how much I can spend
Set up a time out for 24 hours""",
'other': """What's the weather like today?
Tell me a joke
How do I reset my password?
What time does the match start?
Who won the World Cup in 2018?
Can I talk to a human agent?
How do I change my email address?
Thanks, that was helpful
I want to delete my account
Which sports do you cover?
Can you recommend a good restaurant?
Hello there
My app keeps crashing on my phone
How do I verify my identity?
What is the capital of France?
Where can I find your terms and conditions?
Do you have a mobile app for iPhone?
Who is the best player in the world?
What are your customer support hours?
How do I update my address?
Why is the website slow today?
Good morning, how are you?
Can you explain the offside rule?
I forgot my username
Is there a loyalty program?
Who is the manager of Liverpool?
How do I contact support?
Write me a poem about football
Is this service available in Canada?
What is a parlay?""",
}
OUT = Path('data/splits')


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for intent, block in DATA.items():
        lines = [line.strip() for line in block.strip().split('\n') if line.strip()]
        assert len(lines) == 30, (intent, len(lines))
        rows += [{'intent': intent, 'text': text} for text in lines]
    rng = random.Random(13)
    split: dict[str, list[dict]] = {'train': [], 'dev': [], 'test': []}
    for intent in DATA:
        items = [row for row in rows if row['intent'] == intent]
        rng.shuffle(items)
        split['test'] += items[:7]
        split['dev'] += items[7:11]
        split['train'] += items[11:]
    counter = 0
    for name in ['train', 'dev', 'test']:
        rng.shuffle(split[name])
        with open(OUT / f'intents_{name}.jsonl', 'w', encoding='utf-8') as handle:
            for row in split[name]:
                counter += 1
                handle.write(json.dumps({'id': f'intent-{counter:03d}', **row}) + '\n')
        print(name, len(split[name]))


if __name__ == '__main__':
    main()

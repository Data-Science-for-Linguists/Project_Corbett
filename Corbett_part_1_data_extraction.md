
# Reddit Data Trimming
Robert Corbett
rwc27@pitt.edu

Due to the size of the files, I can not provide the raw data files on github.

Due to the size of the files, I can not provie the raw data files on github.  For this example, I provided a file named "sample_data.txt" which is made of the first 2000 json entries in the file for August 2017.  The full August 2017 file is 48.1GB and contains all 84,658,503 posts made to reddit in that month.  For this project, I am hoping to use atleast 1 year of data, so I will need to go through roughly just over 1 billion posts.  When I ran this code of the full August 2017 file, it took my computer about 4 hours to complete.

This notebook is just used to demonstrate how I will collect posts of interest from the larger files. The code was written so I can just point file paths to new locations and run.


```python
import json
```

The first thing I must do for my project is find which subreddit threads I want to look at.  Below, the code iterates through all of the subreddit tags on the json entries.  It adds new unique subreddit tags to the Subreddit list.  When I did this on the full August file, I only checked the first 100,000 tags, which provie almost 10,000 unique tags.  For my project, I am looking for threads that are active.  Half of the threads will be political in nature, while the other half will not (I'm looking for hobbie threads).


```python
x = 0
Subreddit = []
with open("sample_data.txt", 'r') as file_in:
    for line in file_in:
        line_in = json.loads(line)
        if line_in['subreddit'] not in Subreddit:
            Subreddit.append(line_in['subreddit'])
        x = x + 1
        if x == 100000:
            break
print(Subreddit)
```

    ['Filmmakers', 'Addons4Kodi', 'NotTimAndEric', 'Saber', 'The_Donald', 'JordanPeterson', 'AskReddit', 'Military', 'trashy', 'KitchenConfidential', 'TagPro', 'DotA2', 'PuzzleAndDragons', 'SecretWorldLegends', 'SquaredCircle', 'Tinder', 'buccos', 'brasil', 'niceguys', 'Norway', 'd3fecttesting', 'NYYankees', 'acrl', 'airsoft', 'loseit', 'women', 'RoastMe', 'business', 'runescape', 'harrypotter', 'solareclipse', 'SCP', 'techsupport', 'asoiaf', 'MDMA', 'RocketLeague', 'sex', 'Trivium', 'anime', 'politics', 'redsox', 'DebateReligion', 'pcmasterrace', 'Retconned', 'comicbookmovies', 'RocketLeagueExchange', 'Roadcam', 'CatGifs', 'Bondage', 'shittankiessay', 'starbound', 'halifax', 'ADHD', 'pathofexile', 'UpliftingNews', 'teenagers', 'nba', 'sg_playground', 'Games', 'leagueoflegends', 'KingdomHearts', 'BigBrother', 'MobiusFF', 'EnoughTrumpSpam', 'dvdcollection', 'AMA', 'indie_rock', 'HouseOfCards', 'Jokes', 'TheDickShow', 'Fishing', 'WaltDisneyWorld', 'Harmontown', 'ladybonersgw', 'kik', 'BlackPeopleTwitter', 'cscareerquestions', 'streetwear', 'Steam', 'bettafish', 'videos', 'milwaukee', 'pics', 'reddevils', 'Fantasy', 'GoneWildPlus', 'Awwducational', 'atheism', 'MensRights', 'titanfall', 'florida', 'GlobalOffensiveTrade', 'CringeAnarchy', 'iamverybadass', 'LateStageCapitalism', 'AskWomen', 'musicals', 'Teachers', 'AskTrumpSupporters', 'PoliticsWithoutTheBan', 'CrazyIdeas', 'JRPG', 'RepLadies', 'depression', 'battlefield_one', 'Neverwinter', 'nonmonogamy', 'thebachelor', 'incest', 'investing', 'exmormon', 'KotakuInAction', 'announcements', 'MLBTheShow', 'tipofmytongue', 'insurgency', 'hockey', 'vexillology', 'radiocontrol', 'formula1', 'fatlogic', 'Fitness', 'walmart', 'mildlyinteresting', 'darksouls3', 'bjj', 'Sat', 'typewriters', 'BitcoinMarkets', 'movies', 'JUSTNOMIL', 'gameofthrones', 'seinfeld', 'CompulsiveSkinPicking', 'Archaeology', 'INJUSTICE', 'StreamersGoneWild', 'nyjets', 'houston', 'ultimate', 'JurassicPark', 'news', 'AskNYC', 'PokemonGoMPLS', 'confession', 'science', 'ptcgo', 'NASCAR', 'diablo3', 'rpghorrorstories', 'HFY', 'WahoosTipi', 'benchmade', 'forhonor', 'audioengineering', 'gaming', 'Vive', 'nosleep', 'MMA', 'Defcon', 'worldpowers', 'AskAnAmerican', 'gifs', 'oddlysatisfying', 'KCRoyals', 'HaloStory', 'EarthFans', 'Brawlstars', 'footballmanagergames', 'thesopranos', 'nfl', 'asktrp', 'HomemadeGayPorn', 'newzealand', 'rule34', 'explainlikedrcox', 'RateMyNudeBody', 'funny', 'RyzeMains', 'personalfinance', 'Amd', 'UKPersonalFinance', 'rugbyunion', 'hearthstone', 'hiphopheads', 'PlantedTank', 'ketamine', 'DBZDokkanBattle', 'rant', 'watercooling', 'asatru', 'vegan', 'japancirclejerk', 'stlouisblues', 'fantasybaseball', 'penguins', 'splatoon', 'soccer', 'trees', 'hardwareswap', 'jaimebrienne', 'WTF', 'AskOuija', 'starcitizen', 'travel', 'Hoocoodanode', 'counting', 'heroesofthestorm', 'oculus', 'ChildrenFallingOver', 'Denver', 'gamegrumps', 'Libertarian', 'pnwriders', 'Futurism', 'marvelstudios', 'WeWantPlates', 'battlefield_live', 'fatpeoplestories', 'PUBATTLEGROUNDS', 'xboxone', 'AFL', 'guitarpedals', 'IronThronePowers', 'DestinyTheGame', 'GlobalOffensive', 'Philippines', 'Reformed', 'StudentLoans', 'Coffee', 'TTC_PCOS', 'OldSchoolCool', 'Charlotte', 'lawncare', 'ShitPoliticsSays', 'buildapcsales', 'CODZombies', 'rickandmorty', 'sports', 'TheSilphRoad', 'CoDCompPlays', 'hardware', 'changemyview', 'FierceFlow', 'conspiracy', 'exmuslim', 'TexasRangers', 'AskMen', 'PrequelMemes', 'magicTCG', 'Shadowrun', 'ShingekiNoKyojin', 'bladeandsoul', 'todayilearned', 'Aquariums', 'programming', 'explainlikeimfive', 'CHIBears', 'BlackMetal', 'wallstreetbets', 'NoStupidQuestions', 'weddingplanning', 'dating_advice', 'Austin', 'Indiana', 'funkopop', 'RedditSilverRobot', 'Pyongyang', 'smashbros', 'Warthunder', 'ActionFigures', 'rupaulsdragrace', 'PoliticalDiscussion', 'vancouver', '3DS', 'askscience', 'PoliticalHumor', 'CalPoly', 'eatsandwiches', 'LosAngeles', 'EatCheapAndHealthy', 'germany', 'korea', 'AskRedditAfterDark', 'BrantSteele', 'learnmath', 'CrusaderKings', 'ems', 'ESFJ', 'holdmyfries', 'bestof', 'pokemongo', 'guns', 'KyliePage', 'Roast_Me', 'buildapc', 'noveltranslations', 'donaldglover', 'television', 'firefox', 'discgolf', 'EDC', 'woahdude', 'wholesomememes', 'pokemonduel', 'freefolk', 'blender', 'pcgamingtechsupport', 'vzla', 'MachineLearning', 'SubredditSimulator', 'NatureIsFuckingLit', 'Unexpected', 'pokemon', 'INTP', 'howardstern', 'ireland', 'diypedals', 'Drama', 'medical', 'edc_raffle', 'pureasoiaf', 'MtF', 'MakeupAddiction', 'bostonceltics', 'Conservative', 'indieheads', 'yiff', 'CasualConversation', 'DrugStashes', 'TownofSalemgame', 'thatHappened', 'DunderMifflin', 'army', 'casualiama', 'LigaMX', '4chan', 'Warframe', 'Pareidolia', 'evenewbies', 'PSVR', 'FinalFantasy', 'IASIP', 'teslamotors', 'WorldOfWarships', 'SUBREDDITNAME', 'gmod', 'Overwatch', 'ComedyCemetery', 'CEOfriendly', 'ImagesOfromania', 'Realestatefinance', 'Rainmeter', 'toofers', 'mopeio', 'SanctionedSuicide', 'GuitarHero', 'LifeProTips', 'DataHoarder', 'grandrapids', 'Delightfullychubby', 'college', 'Sneakers', 'NewsOfTheStupid', 'wow', 'Morrowind', 'PublicFreakout', 'leafs', 'chile', 'buildapcforme', 'WayOfTheBern', 'nottheonion', 'canada', 'comicbooks', 'legaladvice', 'fashionsouls', 'anime_irl', 'Warhammer40k', 'drummers', 'NoSillySuffix', 'nevertellmetheodds', 'NBA2k', 'ThePeoplesRCigars', 'polandball', 'breakingmom', 'amateurradio', 'RotMG', 'justneckbeardthings', 'unpopularopinion', 'traps', 'fantasybball', 'youngjustice', 'ETHInsider', 'childfree', 'worstepisodeever', 'BullTerrier', 'LastDayonEarthGame', 'ffxiv', 'MemeEconomy', 'HatsuVault', 'askgaybros', 'GCdebatesQT', 'JoeRogan', 'keto', 'howto', 'FFBraveExvius', 'AnimalsBeingBros', 'malefashionadvice', 'OnePiece', 'eldertrees', 'microgrowery', 'MechanicalKeyboards', 'AmateurRoomPorn', 'MorganHultgren', 'Morristown', 'caseyneistat', 'OutOfTheLoop', 'pcgaming', 'AgainstHateSubreddits', 'beer', 'GamerPals', 'stopsmoking', 'ddlg', 'CFB', 'Competitiveoverwatch', 'totalwar', 'iOSProgramming', 'ripcity', 'SeattleWA', 'raisedbynarcissists', 'lonely', 'lewronggeneration', 'Madden', 'IncelTears', 'OkCupid', 'CampHalfBloodRP', 'worldnews', 'LowTierTradingRL', 'Atlanta', 'motorcycles', 'NewSkaters', 'simpleliving', 'quittingkratom', 'FireEmblemHeroes', 'bdsm', 'singedmains', 'Dariusmains', 'metaldetecting', 'Malazan', 'Dota2Trade', 'Eve', 'RateMyMayor', 'opieandanthony', 'TrumpCriticizesTrump', 'Marvel', 'Justrolledintotheshop', 'ABraThatFits', 'arrow', 'SanJose', 'halloween', '40kLore', 'lifeisstrange', 'melbourne', 'economy', 'singapore', 'FFRecordKeeper', 'DIY_eJuice', 'survivor', 'OnePieceTC', 'cowboys', 'IMDbFilmGeneral', 'DNMUK', 'Torontobluejays', 'MkeBucks', 'BeautyBoxes', 'nakedladies', 'KingkillerChronicle', 'Advice', 'photography', 'Bitcoin', 'RandomActsOfGaming', 'gonewild', 'Tekken', 'electronic_cigarette', 'CrappyDesign', 'writing', 'DirtySnapchat', 'mallninjashit', 'MrRobot', 'intj', 'leaves', 'F13thegame', 'Career_Advice', 'AnimalsBeingJerks', 'france', 'Cigarettes', 'gamindustri', 'suggestmeabook', 'FloridaGators', 'aww', 'Flute', 'Showerthoughts', 'TrollXChromosomes', 'worldbuilding', 'Goruck', 'fakeid', 'linux', 'Brewers', 'PoGoDFW', 'SubredditDrama', 'eu4', 'ChoosingBeggars', 'iamverysmart', 'badwomensanatomy', 'flatearth', 'Rainbow6', 'nrl', 'dankmemes', 'indianpeoplefacebook', 'sydney', 'Christianity', 'fordranger', 'Guildwars2', 'BitcoinAll', 'FemBoys', 'russian', 'ThriftStoreHauls', 'short', 'relationship_advice', 'Infinitewarfare', 'speedrun', 'UMD', 'pocketrumble', 'Nioh', 'gtaonline', 'JapanTravel', 'usanews', 'dirtypenpals', 'Berserk', 'Indiemakeupandmore', 'blackpeoplegifs', 'HardwareSwapUK', 'ak47', 'shakespeare', 'stopdrinking', 'MMORPG', 'quotes', 'gonewild30plus', 'uber', 'Frat', 'twinpeaks', 'Kaiserreich', 'DecemberBumpers2017', 'FIFA', 'Staples', 'swgoh_guilds', 'me_irl', 'SubaruForester', 'interestingasfuck', 'CitiesSkylines', '2007scape', 'canucks', 'supremeclothing', 'future_fight', 'CHICubs', 'thedivision', 'FoundPaper', 'grandorder', 'cringe', 'Catholicism', 'poker', 'australia', 'WredditCountryClub', 'exjw', 'doctorwho', 'NoMansSkyTheGame', 'asstastic', 'tattoos', 'baseball', 'sadcringe', 'whitesox', 'gaybros', 'youtubecomments', 'lawschooladmissions', 'WritingPrompts', 'asktransgender', 'DynastyFF', 'uwaterloo', 'sousvide', 'geopolitics', 'nyc', 'Drugs', 'slaterefugees', 'JizzedToThis', 'milliondollarextreme', 'Bobbers', 'MouseReview', 'redditgetsdrawn', 'ImagesOfCanada', 'NHLHUT', 'FinalFantasyTCG', 'ProgrammerHumor', 'blackops3', 'DuelLinks', 'NMSGalacticHub', 'Android', 'TryingForABaby', 'opiates', 'tucker_carlson', 'flightsim', 'RWBY', 'elderscrollsonline', 'StarWarsBattlefront', 'Texans', 'boston', 'Serendipity', 'makeupexchange', 'RealGirls', 'NoFapChristians', 'shittyrainbow6', 'Futurology', 'Trove', 'dirtykikpals', 'argentina', 'Random_Acts_Of_Amazon', 'NintendoSwitch', 'EngineeringStudents', 'keming', 'askfuneraldirectors', 'NoFap', 'fo4', 'androidapps', 'ethereum', 'gonewildcurvy', 'CapitalismVSocialism', 'orioles', 'arcadefire', 'nextdaysurvival', 'ContestOfChampions', 'bigdickproblems', 'GooglePixel', 'GamerGhazi', 'electronics', 'MakeupRehab', 'Ubiq', 'EliteDangerous', 'Syracuse', 'Ohio', 'fantasyfootball', 'Parkour', 'TrueFMK', 'Mordhau', 'blackdesertonline', 'HealthInsurance', 'ExpandDong', 'theydidthemath', 'boardgames', 'foxholegame', 'longrange', 'H3VR', 'btc', 'wowservers', 'portugal', 'sandiego', 'Shadowverse', 'ImagesOfAustralia', 'photoshopbattles', 'namenerds', 'DC_Cinematic', 'dogs', 'thesims', 'skeptic', 'kratom', 'furry_irl', 'kpop', 'playrust', 'GTAV', 'synthesizers', 'CallOfDuty', 'NeverTrump', 'playark', 'whatisthisthing', 'TumblrInAction', 'Glocks', 'forwardsfromgrandma', 'canadaguns', 'GameSale', 'UnresolvedMysteries', 'ExploreFiction', '2meirl4meirl', 'FashionReps', 'relationships', 'fountainpens', 'Quebec', 'GetMotivated', 'facebookwins', 'treephilly', 'EscapefromTarkov', 'BannedFromThe_Donald', 'DebateCommunism', 'SweatyPalms', 'Repsneakers', 'watchpeopledie', 'Steroidsourcetalk', 'DCComicsLegendsGame', 'maryland', 'ElectricSkateboarding', 'psg', 'shortscarystories', 'FocusST', 'ShitWehraboosSay', 'mexico', 'ABDL', 'starterpacks', 'AnnePro', 'patientgamers', 'C_S_T', 'forza', 'horror', 'popping', 'fivenightsatfreddys', 'spotted', 'Cumtown', 'redacted', 'JonTron', 'shylittlesunflower', 'battlefield_4', 'offmychest', 'pipemaking', 'nursing', 'OfficeDepot', 'Ripple', 'flying', 'ForeverAlone', 'rollercoasters', 'Anticonsumption', 'polyamory', 'sanfrancisco', 'LifeRPG', 'Guiltygear', 'cars', 'orangetheory', 'sales', 'drupal', 'DebateAChristian', 'summonerschool', 'rust', 'TapTitans2', 'ApplyingToCollege', 'PS4Deals', 'tf2', 'MST3K', 'Woodcarving', 'dndnext', 'europe', 'socialskills', 'Unity3D', 'Miata', 'transpositive', 'AskThe_Donald', 'kindafunny', 'TheRedPill', 'reactiongifs', 'WeAreTheMusicMakers', 'TokyoGhoul', 'Grimdank', 'Smite', 'nasusmains', 'Ducati', 'DeFranco', 'starshiptroopers', 'mildlyinfuriating', 'CryptoCurrency', 'Denmark', 'rotmgprojectb', 'seduction', 'AmateurWifes', 'skyrim', 'AmericanHorrorStory', 'photomarket', 'nintendo', 'homelabsales', 'shockwaveporn', 'avengersacademygame', 'motorcitykitties', 'YGOSales', 'secretsubgonewild', 'Mcat', 'BurningMan', 'thewallstreet', 'neoliberal', 'gwent', 'TwoBestFriendsPlay', 'FellowKids', 'EmmaWatson', 'battlebots', 'philadelphia', 'madlads', 'headphones', 'OnceUponAnEnd', 'redheads', 'RetributionOfScyrah', 'photoshop', 'TenYearsAgoOnReddit', 'BBQ', 'Winnipeg', 'mixedrace', 'wholesomebpt', 'ACT', 'YouEnterADungeon', 'CSRRacing2', 'Watches', 'StarWars', 'replications', 'FUTMobile', 'AskEurope', 'StarTrekContinues', 'LabourUK', 'roadtrip', 'pharmacy', 'CasualUK', 'ledgerwallet', 'MovieDetails', 'cocktails', 'Bass', 'HaircareScience', 'PC_Builders', 'AssholeBehindThong', 'FiveYearsAgoOnReddit', 'transformers', 'infertility', 'williamandmary', 'HondaCB', 'AccidentalWesAnderson', 'ethtrader', 'amiugly', 'verizon', 'paragon', 'Stellaris', 'Ice_Poseidon', 'retrogaming', 'watch_dogs', 'japan', 'panthers', 'TIL_Uncensored', 'Monero', 'Mario', 'math', 'CFA', 'canadagunsEE', 'SFGiants', 'AsiansGoneWild', 'newsokur', 'CLG', 'EntExchange', 'XWingTMG', 'RedPillWomen', 'Megaten', 'iOSthemes', 'churning', 'mechmarket', 'skateboarding', 'chicago', 'retrobattlestations', 'HuntsvilleAlabama', 'chibike', 'rpg', 'origin', 'Cyberpunk', 'puppy101', 'answers', 'SchoolIdolFestival', 'CalamariRaceTeam', 'fullmoviesonanything', 'evangelion', 'Mariners', 'ClashOfClans', 'bicycling', 'AbletonProductions', 'starcraft', 'HumansBeingBros', 'nSuns', 'cigars', 'Vault_Tec_Corporation', 'GalaxyS8', 'progmetal', 'IAmA', 'HeistTeams', 'cookingforbeginners', 'Hulu', 'LetItDie', 'skrillex', 'borussiadortmund', 'Portland', 'brisbane', 'Patriots', 'memes', 'gunpolitics', 'AntiJokes', 'Jeopardy', 'MarchAgainstTrump', 'esist', 'ANI_COMMUNISM', 'YGOBinders', 'Dodgers', 'Miniswap', 'ImagesOfArgentina', 'walkingwarrobots', 'TheOriginals', 'SketchDaily', 'Helldivers', 'ukpolitics', 'GameDeals', 'greece', 'MakeupAddictionCanada', 'MasterofNone', 'Enough_Sanders_Spam', '3Dprinting', 'DCcomics', 'food', 'kancolle', 'espnyankees', 'CCW', 'longisland', '1200isplenty', 'apple', 'rawdenim', 'kickstarter', 'TOtrees', 'LipsThatGrip', 'Intelligence', 'AndroidMasterRace', 'Injustice2MobileGame', 'Tiki', 'Mustang', 'CrohnsDisease', '3dspiracy', 'masseffect', 'trashyboners', 'AdviceAnimals', 'giantbomb', 'MTGLegacy', 'hockeyplayers', 'G502MasterRace', 'Guitar', 'GoneMild']
    


```python
print(len(Subreddit))
```

    946
    

The list of 2,000 entries managed to contain 946 unique subreddit threads.  I now need to extract the threads I am interested in from the larger file.  I did this by creating an empty list for each thread I want to extract.  I then needed to loop through each line in the json file and compare the subreddit tag to the string I am looking for.  If the tag is found, I append the line to the appropriate list.  If the tag is not found, I add one to the variable 'other' so I can later make sure that every json entry was checked.


```python
x = 0
The_Donald = []
politics = []
indie_rock = []
AskTrumpSupporters = []
PoliticsWithoutTheBan = []
hockey = []
seinfeld = []
Archaeology = []
Libertarian = []
Futurism = []
Conservative = []
indieheads = []
worldnews = []
photography = []
geopolitics = []
Android = []
boardgames = []
NeverTrump = []
neoliberal = []
headphones = []
gunpolitics = []
MarchAgainstTrump = []
DCcomics = []
food = []
other = 0
with open("sample_data.txt", 'r') as file_in:
    for line in file_in:
        line_in = json.loads(line)
        subreddit = line_in['subreddit']
        if subreddit == 'The_Donald':
            The_Donald.append(line_in)
        elif subreddit == 'politics':
            politics.append(line_in)
        elif subreddit == 'indie_rock':
            indie_rock.append(line_in)
        elif subreddit == 'AskTrumpSupporters':
            AskTrumpSupporters.append(line_in)
        elif subreddit == 'PoliticsWithoutTheBan':
            PoliticsWithoutTheBan.append(line_in)
        elif subreddit == 'hockey':
            hockey.append(line_in)
        elif subreddit == 'seinfeld':
            seinfeld.append(line_in)
        elif subreddit == 'Archaeology':
            Archaeology.append(line_in)
        elif subreddit == 'Libertarian':
            Libertarian.append(line_in)
        elif subreddit == 'Futurism':
            Futurism.append(line_in)
        elif subreddit == 'Conservative':
            Conservative.append(line_in)
        elif subreddit == 'indieheads':
            indieheads.append(line_in)
        elif subreddit == 'worldnews':
            worldnews.append(line_in)
        elif subreddit == 'photography':
            photography.append(line_in)
        elif subreddit == 'geopolitics':
            geopolitics.append(line_in)
        elif subreddit == 'Android':
            Android.append(line_in)
        elif subreddit == 'boardgames':
            boardgames.append(line_in)
        elif subreddit == 'NeverTrump':
            NeverTrump.append(line_in)
        elif subreddit == 'neoliberal':
            neoliberal.append(line_in)
        elif subreddit == 'headphones':
            headphones.append(line_in)
        elif subreddit == 'gunpolitics':
            gunpolitics.append(line_in)
        elif subreddit == 'MarchAgainstTrump':
            MarchAgainstTrump.append(line_in)
        elif subreddit == 'DCcomics':
            DCcomics.append(line_in)
        elif subreddit == 'food':
            food.append(line_in)
        else:
            other = other +1
        x = x + 1
        #if x == 100000:   allows me to iterate through a portion of the file instead of the whole file
        #    break
```

Next, I checked how many posts I retrieved from each subreddit.  The variable x tracks how many posts I extracted so I can add it to the 'other' variable so I can make sure all posts where checked.


```python
x = 0
print('The_Donald: ' + str(len(The_Donald)))
x = x + len(The_Donald)

print('politics: ' + str(len(politics)))
x = x + len(politics)

print('indie_rock: ' + str(len(indie_rock)))
x = x + len(indie_rock)

print('AskTrumpSupporters: ' + str(len(AskTrumpSupporters)))
x = x + len(AskTrumpSupporters)

print('PolitisWithoutTheBan: ' + str(len(PoliticsWithoutTheBan)))
x = x + len(PoliticsWithoutTheBan)

print('hockey: ' + str(len(hockey)))
x = x + len(hockey)

print('seinfeld: ' + str(len(seinfeld)))
x = x + len(seinfeld)

print('Archaeology: ' + str(len(Archaeology)))
x = x + len(Archaeology)

print('Libertarian: ' + str(len(Libertarian)))
x = x + len(Libertarian)

print('Futurism: ' + str(len(Futurism)))
x = x + len(Futurism)

print('Conservative: ' + str(len(Conservative)))
x = x + len(Conservative)

print('indie_heads: ' + str(len(indieheads)))
x = x + len(indieheads)

print('worldnews: ' + str(len(worldnews)))
x = x + len(worldnews)

print('photography: ' + str(len(photography)))
x = x + len(photography)

print('geopolitics: ' + str(len(geopolitics)))
x = x +len(geopolitics)

print('Android: ' + str(len(Android)))
x = x + len(Android)

print('boardgames: ' + str(len(boardgames)))
x = x + len(boardgames)

print('NeverTrump: ' + str(len(NeverTrump)))
x = x + len(NeverTrump)

print('neoliberal: ' + str(len(neoliberal)))
x = x + len(neoliberal)

print('headphones: ' + str(len(headphones)))
x = x + len(headphones)

print('gunpolitics: ' + str(len(gunpolitics)))
x = x + len(gunpolitics)

print('MarchAgainstTrump: ' + str(len(MarchAgainstTrump)))
x = x + len(MarchAgainstTrump)

print('DCcomics: ' + str(len(DCcomics)))
x = x + len(DCcomics)

print('food: ' + str(len(food)))
x = x + len(food)

print('total: ' + str(x))
print('check sum: ' + str(x + other))
```

    The_Donald: 22
    politics: 67
    indie_rock: 1
    AskTrumpSupporters: 4
    PolitisWithoutTheBan: 1
    hockey: 4
    seinfeld: 1
    Archaeology: 1
    Libertarian: 4
    Futurism: 1
    Conservative: 2
    indie_heads: 1
    worldnews: 15
    photography: 2
    geopolitics: 1
    Android: 4
    boardgames: 4
    NeverTrump: 1
    neoliberal: 1
    headphones: 1
    gunpolitics: 1
    MarchAgainstTrump: 1
    DCcomics: 1
    food: 1
    total: 142
    check sum: 2000
    

As you can see above, from the 2000 entries I used in this example, I found 142  entries from the 24 threads I was looking for.  When I ran this on the August 2017 file, I was able to shrink that 48.1GB file down to 24 files that combined were 2.83GB.

I then need to save the lists to a file so I can use them later.  I created a file to hold the months data and then need to create and write to a file for each list.  I use the json.dump() function to add each json entry as a line in the text file and then need to write a new line character to seperate them.


```python
s = 'August_17'

with open(s + '/The_Donald.txt', 'w') as outfile:
    for item in The_Donald:
        json.dump(item, outfile)
        outfile.write("\n")
    
with open(s + '/politics.txt', 'w') as outfile:
    for item in politics:
        json.dump(item, outfile)
        outfile.write("\n")
    
with open(s + '/indie_rock.txt', 'w') as outfile:
    for item in indie_rock:
        json.dump(item, outfile)
        outfile.write("\n")
    
with open(s + '/AskTrumpSupporters.txt', 'w') as outfile:
    for item in AskTrumpSupporters:    
        json.dump(item, outfile)
        outfile.write("\n")
    
with open(s + '/PoliticsWithoutTheBan.txt', 'w') as outfile:
    for item in PoliticsWithoutTheBan:
        json.dump(item, outfile)
        outfile.write("\n")

with open(s + '/hockey.txt', 'w') as outfile:
    for item in hockey:    
        json.dump(item, outfile)
        outfile.write("\n")
    
with open(s + '/seinfeld.txt', 'w') as outfile:
    for item in seinfeld:    
        json.dump(item, outfile)
        outfile.write("\n")
    
with open(s + '/Archaeology.txt', 'w') as outfile:
    for item in Archaeology:    
        json.dump(item, outfile)
        outfile.write("\n")
    
with open(s + '/Libertarian.txt', 'w') as outfile:
    for item in Libertarian:    
        json.dump(item, outfile)
        outfile.write("\n")
    
with open(s + '/Futurism.txt', 'w') as outfile:
    for item in Futurism:
        json.dump(item, outfile)
        outfile.write("\n")
        
with open(s + '/Conservative.txt', 'w') as outfile:
    for item in Conservative:
        json.dump(item, outfile)
        outfile.write("\n")
        
with open(s + '/indieheads.txt', 'w') as outfile:
    for item in indieheads:
        json.dump(item, outfile)
        outfile.write("\n")
    
with open(s + '/worldnews.txt', 'w') as outfile:
    for item in worldnews:
        json.dump(item, outfile)
        outfile.write("\n")
        
with open(s + '/photography.txt', 'w') as outfile:
    for item in photography:
        json.dump(item, outfile)
        outfile.write("\n")
    
with open(s + '/geopolitics.txt', 'w') as outfile:
    for item in geopolitics:
        json.dump(item, outfile)
        outfile.write("\n")
        
with open(s + '/Android.txt', 'w') as outfile:
    for item in Android:
        json.dump(item, outfile)
        outfile.write("\n")
        
with open(s + '/boardgames.txt', 'w') as outfile:
    for item in boardgames:
        json.dump(item, outfile)
        outfile.write("\n")
        
with open(s + '/NeverTrump.txt', 'w') as outfile:
    for item in NeverTrump:
        json.dump(item, outfile)
        outfile.write("\n")
        
with open(s + '/neoliberal.txt', 'w') as outfile:
    for item in neoliberal:    
        json.dump(item, outfile)
        outfile.write("\n")
        
with open(s + '/headphones.txt', 'w') as outfile:
    for item in headphones:
        json.dump(item, outfile)
        outfile.write("\n")
        
with open(s + '/gunpolitics.txt', 'w') as outfile:
    for item in gunpolitics:
        json.dump(item, outfile)
        outfile.write("\n")
        
with open(s + '/MarchAgainstTrump.txt', 'w') as outfile:
    for item in MarchAgainstTrump:
        json.dump(item, outfile)
        outfile.write("\n")
    
with open(s + '/DCcomics.txt', 'w') as outfile:
    for item in DCcomics:
        json.dump(item, outfile)
        outfile.write("\n")
        
with open(s + '/food.txt', 'w') as outfile:
    for item in food:    
        json.dump(item, outfile)
        outfile.write("\n")
```

The program took several hours to run. I wanted to use timeit but when I looked into it, I found that timeit turns off automatic garbage collection and I was afraid with such a large file, that it might cause problems with memory usage.

After I run this program, I am left with smaller more manageable files that are still in json format.  In the next part, I will open these files and create dataframes that I am able to explore and manipulate.


```python
#x = 0
#list_range = range(len(lines))
#for i in list_range:
#    foo = (lines[i - 1])
#    beg = 0
#    while beg != -1:
#        beg = foo.find('apple', beg + 1)
#        if beg != -1:
#            print("line: " + str(i) + "    " + foo + " \n")
#            x = x + 1
#print(x)
```

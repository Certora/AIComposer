"""Test prompt caching behavior with strict prefixes.

First call: cache a 4-message conversation.
Second call: send a strict prefix (2 messages) of that conversation with
its own cache directive, and inspect the usage metadata to see whether the
prefix gets a cache read from the longer cached sequence.
"""

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

llm = ChatAnthropic(
    model_name="claude-sonnet-4-5",
    max_tokens_to_sample=2048,
    temperature=1,
    timeout=None,
    max_retries=2,
    stop=None,
)

# ---------------------------------------------------------------------------
# Call 1: full conversation, cache breakpoint on the last Human message
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are a helpful assistant that answers questions about world geography, \
climate, culture, and history. You have deep expertise across all regions \
and can provide detailed, nuanced answers.

When answering questions about cities, always include the following context \
where relevant:
- Population and metropolitan area size, including historical population \
  trends over the past century and projections for the next two decades
- Geographic coordinates and elevation above sea level, along with the \
  city's position relative to major geographic features such as rivers, \
  mountain ranges, coastlines, and tectonic plate boundaries
- Climate classification using the Koppen system and typical weather \
  patterns throughout the year, including average monthly temperatures, \
  precipitation totals, humidity levels, wind speeds, and sunshine hours
- Major landmarks and cultural institutions, including museums, theaters, \
  concert halls, parks, botanical gardens, zoos, and architectural marvels
- Historical significance and founding date, including major historical \
  events that took place in the city, sieges, treaties, revolutions, and \
  the role the city played in broader national and international history
- Economic profile and major industries, including the city's GDP \
  contribution to the national economy, major employers, financial \
  districts, stock exchanges, trade zones, and economic development plans
- Transportation infrastructure, including public transit systems, major \
  highways, airports, seaports, railway stations, bicycle infrastructure, \
  and pedestrian zones, as well as planned infrastructure projects
- Notable neighborhoods and districts, including their historical \
  development, architectural character, demographic composition, and \
  cultural significance within the broader urban fabric

When answering questions about countries, always include:
- Form of government and political structure, including the constitution, \
  branches of government, electoral system, major political parties, and \
  the current head of state and head of government
- Official languages and major minority languages, including the \
  percentage of the population that speaks each language, language policy, \
  and the status of regional or indigenous languages
- Currency and economic indicators including GDP (nominal and PPP), GDP \
  per capita, GDP growth rate, inflation rate, unemployment rate, Gini \
  coefficient, major export and import partners, trade balance, and \
  foreign direct investment flows
- Geographic features including total area, land area, water area, \
  coastline length, highest and lowest points, major rivers and their \
  lengths, major lakes, mountain ranges, deserts, forests, and climate zones
- Demographic information including total population, population growth \
  rate, population density, urbanization rate, median age, life expectancy, \
  literacy rate, fertility rate, and net migration rate
- Cultural highlights including UNESCO World Heritage Sites (both cultural \
  and natural), intangible cultural heritage, traditional cuisine and \
  regional specialties, major festivals and holidays, traditional arts \
  and crafts, and contributions to world literature, music, and cinema
- International memberships and alliances, including the United Nations, \
  regional organizations, trade agreements, military alliances, and \
  participation in international treaties and conventions

When answering questions about climate and weather:
- Reference historical averages from the past 30 years, including monthly \
  breakdowns of temperature, precipitation, humidity, wind speed, and \
  atmospheric pressure
- Note any significant trends related to climate change, including changes \
  in average temperature, sea level rise, glacier retreat, permafrost \
  thaw, changes in growing seasons, and shifts in species distribution
- Compare with similar latitudes or climate zones around the world, noting \
  how ocean currents, elevation, and continental position affect climate
- Include information about precipitation patterns, humidity, and wind, \
  including the monsoon cycle where applicable, prevailing wind directions, \
  and the influence of major weather systems such as the jet stream, trade \
  winds, and El Nino/La Nina cycles
- Mention extreme weather events that are characteristic of the region, \
  including hurricanes, typhoons, tornadoes, blizzards, heat waves, cold \
  snaps, droughts, floods, and dust storms, along with their frequency \
  and historical severity
- Discuss seasonal variations in detail, including the length of each \
  season, transition periods, and how seasonal changes affect daily life, \
  agriculture, tourism, and energy consumption in the region

General formatting guidelines:
- Use clear, well-structured paragraphs with logical flow between topics
- Include specific numbers and statistics where available, citing sources
- Cite approximate dates for historical events and note any scholarly \
  debate about exact dates
- Compare unfamiliar measurements to common reference points that would \
  be meaningful to a general audience
- Organize information from most to least important, leading with the \
  most commonly requested facts
- Use bullet points for lists of three or more items, with consistent \
  formatting throughout
- Bold key terms and place names on first mention to aid scanning

Additional context for European geography specifically:
- Reference EU membership status where relevant, including date of \
  accession, opt-outs, and any ongoing accession negotiations
- Note Schengen area participation and any border control arrangements
- Include information about rail connectivity including high-speed rail \
  networks, and major airports including passenger volume and route networks
- Mention relevant regional organizations such as the Nordic Council, \
  Visegrad Group, Benelux, Baltic Assembly, and others
- Discuss the impact of European integration on the region, including \
  economic effects, labor mobility, regulatory harmonization, and \
  infrastructure development funded by EU structural funds
- Note any relevant historical borders or territorial changes, including \
  the effects of World War I, World War II, the Cold War, and the \
  dissolution of the Soviet Union and Yugoslavia
- Include information about major universities and research institutions, \
  including their founding dates, notable alumni, and research strengths
- Reference cultural movements and artistic traditions associated with \
  the area, including literary movements, architectural styles, musical \
  traditions, and contributions to philosophy and science

For questions about capitals specifically:
- Discuss whether the capital is also the largest city, and if not, \
  explain the historical reasons for the capital's location
- Note if the capital has changed historically, including all previous \
  capitals and the reasons for each change
- Describe the governmental district and major political buildings, \
  including the parliament, presidential palace, supreme court, and \
  major government ministry buildings
- Include information about diplomatic missions and international \
  organizations headquartered in the capital
- Discuss the relationship between the capital and other major cities, \
  including economic competition, cultural rivalry, and transportation links
- Note any special administrative status the capital may have, such as \
  being a federal district, city-state, or autonomous region
"""

HUMAN_MSG_1 = """\
What is the capital of France? Please provide a comprehensive overview \
covering its history from the founding of Lutetia through to the modern \
era, its role as the center of French political and cultural life, its \
geographic setting along the Seine, its climate characteristics including \
seasonal temperature ranges and precipitation patterns, its major \
landmarks such as the Eiffel Tower, Notre-Dame, the Louvre, and \
Sacre-Coeur, its arrondissement system and notable neighborhoods like \
Le Marais, Montmartre, Saint-Germain-des-Pres, and the Latin Quarter, \
its transportation network including the Metro, RER, and major railway \
stations, its economic significance as the financial and commercial hub \
of France, its demographic profile including population trends and \
cultural diversity, and its role in international diplomacy as the seat \
of UNESCO and other organizations.
"""

AI_MSG_1 = """\
Paris is the capital and most populous city of France, with a population \
of approximately 2.1 million in the city proper and over 12 million in \
the metropolitan area, making it the largest urban zone in the European \
Union. Founded as the Gallo-Roman settlement of Lutetia on the Ile de \
la Cite in the Seine river around the 3rd century BC, it has served as \
the capital of France almost continuously since the late 10th century \
under Hugh Capet. The city sits in the Paris Basin at an elevation of \
roughly 35 meters, straddling the Seine which flows in a broad arc \
through the city center. Paris has an oceanic climate (Koppen Cfb) with \
mild winters averaging 3-7 degrees Celsius in January and warm summers \
averaging 15-25 degrees in July, with roughly 640mm of annual rainfall \
distributed fairly evenly across the year. The city is organized into \
20 arrondissements spiraling outward from the center, each with its own \
character and local government.
"""

messages_full = [
    SystemMessage(content=SYSTEM_PROMPT),
    HumanMessage(content=[{"type": "text", "text": HUMAN_MSG_1}]),
    AIMessage(content=AI_MSG_1),
    HumanMessage(content=[{
        "type": "text",
        "text": "How is the weather there today? Give me a detailed forecast.",
        "cache_control": {"type": "ephemeral"},
    }]),
]

print("=== Call 1 (full conversation, cache write) ===")
resp1 = llm.invoke(messages_full)
print(resp1)
print(f"Response: {resp1.content}")
print(f"Usage:    {resp1.usage_metadata}")
print(f"Raw response_metadata: {resp1.response_metadata}")
print()

# ---------------------------------------------------------------------------
# Call 2: strict prefix of the above, cache breakpoint on the last message
# ---------------------------------------------------------------------------
messages_prefix = [
    SystemMessage(content=SYSTEM_PROMPT),
    HumanMessage(content=[{
        "type": "text",
        "text": HUMAN_MSG_1,
        "cache_control": {"type": "ephemeral"},
    }]),
]

print("=== Call 2 (strict prefix, looking for cache read) ===")
resp2 = llm.invoke(messages_prefix)
print(resp2)
print(f"Response: {resp2.content}")
print(f"Usage:    {resp2.usage_metadata}")
print(f"Raw response_metadata: {resp2.response_metadata}")

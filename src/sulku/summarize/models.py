from pydantic import BaseModel, ConfigDict, Field


class StyleVector(BaseModel):
    """
    The style axis dictates the mechanical and linguistic construction of an article, focusing on elements like vocabulary complexity, sentence length, and syntax. Writers calibrate this axis to control the piece's readability and aesthetic rhythm, ensuring the language practically supports the chosen tone and aligns with the target audience's comprehension level.
    """

    model_config = ConfigDict(use_attribute_docstrings=True)

    tone: str
    """
    The attitude the writer or publication conveys toward the subject matter.

    If perspective is who is looking, and audience is who is reading, tone is how the delivery sounds.

    Examples:
     - Clinical / Detached – Flat, emotionless, and purely factual.
     - Authoritative / Analytical – Confident, educational, and serious.
     - Urgent / Alarmist – Fast-paced, high-stakes, and tension-driven.
     - Empathetic / Poignant – Sensitive, reflective, and human-centric.
     - Conversational / Irreverent – Casual, colloquial, and occasionally sarcastic or witty.
    """

    perspective: str
    """
    Grammatical point of view (who is telling the story), e.g., 'first-person', 'second-person', 'third-person'.

    This is the literal lens through which the story is narrated.

    Examples:
     - Third-Person Objective ("The Invisible Observer")
     - First-Person ("The Participant/Witness")
     - Second-Person ("The Direct Address")
     - Third-Person Narrative/Deep ("The Novelist")
    """

    angle: str
    """
    The Journalistic Angle of the article.

    Even in strict third-person objective writing, the perspective shifts based on which stakeholders the journalist chooses to center.

    Examples:
     - Top-Down (Institutional Perspective)
     - Bottom-Up (Human Interest/Grassroots Perspective)
     - Macroscopic/Analytical (The Systemic Perspective)
    """

    audience: list[str] = Field(..., description="The intended audience of the article")
    """
    The audience axis dictates what background information needs explaining, how the stakes are framed, and why the story matters to the reader.

    Examples of segments:
     - Proximity (The Geographic Audience) – This dictates the scale of the impact and how much local context the writer can assume.
     - Expertise – This dictates the vocabulary used, the depth of the analysis, and the assumed baseline of prior knowledge.
     - Intent – This dictates the structure of the article based on what the reader is trying to achieve.
     - The Affinity Axis – The framing of the facts and the assumed shared values between the writer and the reader.

    Examples of proximity:
     - Hyper-local / Local Audience
     - National Audience
     - Global / International Audience

    Examples of expertise:
     - General / Mass Public
     - Trade / Specialist Audience
     - Academic / Scholarly Audience
     - Niche / Enthusiast Audience

    Examples of intent:
     - The "Actor" (Action-Oriented Audience)
     - The "Observer" (Curiosity-Oriented Audience)

    Examples of affinity:
     - The Broad Consensus (Apolitical/Centrist Audience)
     - The Partisan/Ideological Audience
    """

    type: str
    """
    The structural purpose of the piece.

    Different types of articles use entirely different structures to serve distinct functions for the reader.

    Examples:
     - Hard News / Breaking News – puts the most vital information in the very first sentence (the lead) and tapers off into background details.
     - News Analysis – A piece that steps away from what just happened to focus entirely on why it happened and how it works. It rarely breaks new facts.
     - Feature / Longform – A narrative-driven story. Features use the tools of fiction—scene-setting, character development, dialogue, and sensory details—to report non-fiction.
     - Investigative Journalism – Deep-dive reporting that uncovers concealed truths. These pieces often take months or years to produce and rely on leaked documents, data analysis, and whistleblowers.
     - Opinion / Editorial (Op-Ed) – A subjective argument. Unlike hard news, the writer's personal stance is the entire point. Editorials represent the publication's institutional view, while columns and op-eds represent the individual writer.
     - Service Journalism – "News you can use." It is highly practical, actionable information formatted as guides, lists, or Q&As.
     - Review – A subjective evaluation of a piece of art, entertainment, or consumer product.
     - Press Releases – A strategic document written by an organization, corporation, or government entity and distributed to journalists. It is essentially a pre-packaged news story, typically promotional in nature.
     - Corrections & Editor's Notes – A post-publication amendment that addresses a factual error, misquote, or ethical lapse in a previously published piece.
     - Advertisements & Sponsored Content – Paid content that is designed to look like editorial content.
     - Announcements & Notices – Brief, highly formatted bulletins—such as wedding announcements, real estate transfers, public hearing dates, or basic obituaries.
     - Letters to the Editor – A reader-submitted response to a previously published piece. These are typically short, opinionated, and highly reactive.
     - Live Blog / Live Feed – A rolling stream of updates on a developing, ongoing event.
    """


class ArticleSummary(BaseModel):
    model_config = ConfigDict(use_attribute_docstrings=True)

    headline: str = Field(
        ...,
        description="The headline of the article.",
        examples=[
            "The association, which received millions in subsidies, kept the accounts by hand and paid wages in cash – Stea reported to the police",
            "In Outokumpu, a way to root out urban seagull problems was found – the restoration of islets attracts birds back to the lake",
            "Tapio Tiainen built a giant sailboat for 14 years: “I don’t even want to say how much money has gone”",
            "Löylyyn vai lenkille? Molemmista saa samat hyödyt, kertoo uusi Jyväskylän yliopiston tutkimus",
        ],
    )
    summary: list[str] = Field(
        ...,
        description="A list of summary sentences for the article.",
        examples=[
            [
                "Stea will charge back approximately 455 000 euros from KRIS-Oulu ry. The association helped people with criminal background, but misused the aid.",
                "A request for investigation has been made to the police.",
                "The former executive director of the association denies the abuses.",
                "The association has been filed for bankruptcy. The government is unlikely to get its money back.",
            ],
            [
                "The gull islets restored in Sysmäjärvi in Outokumpu help seagulls and declining waterfowl species.",
                "Metsähallitus removed vegetation from the islets. The seagulls came back and the waterfowl followed.",
                "Waterfowl take shelter in the seagulls. Gulls effectively expel predators.",
                "Rehabilitation could attract seagulls from cities back to the lakes.",
            ],
            [
                "Tapio Tiainen from Korpilahti has been building a sailboat for 14 years. The boat was transported to Loviisa harbour on Friday.",
                "The boat is almost 16 meters long, five meters wide and weighs 30 tons. It is reportedly the largest sailboat built in Central Finland.",
                "Tiainen goes sailing with her friend to the world right after the summer. The first is Portugal.",
            ],
            [
                "Jyväskylän yliopiston tutkimus osoittaa, että saunan ilmankosteus nostaa sykettä ja kehon lämpötilaa.",
                "Löydös on merkittävä, sillä aiemmissa saunatutkimuksissa ilmankosteutta ei ole mitattu tarkasti.",
                "Terveyshyötyjä voi saada matalammilla lämpötiloilla, jos ilmankosteus on korkeampi.",
                "Kostealla iholla hiki ei haihdu. Saunominen vaikuttaa sydämeen samalla tavalla kuin reipas kävely.",
            ],
        ],
    )
    style: StyleVector = Field(..., description="The style vector of the article.")

import os

os.environ.setdefault("USER_AGENT", "pcfitment-chatbot/1.0")

import bs4
from langchain_community.document_loaders import WebBaseLoader

BASE = "https://partsconnect.zendesk.com/hc/en-us/articles/"

# category -> list of article URLs
CATEGORIES = {
    "Getting Started": [
        BASE + "115001118751-Signing-Up",
        BASE + "115001121912-Logging-in-for-the-first-time",
        BASE + "115001122292-Dashboard-Your-Fitment-Info",
        BASE + "10275291602189-Brand-Add-Use-Brands",
        BASE + "115001118571-How-to-Add-parts",
        BASE + "39964741481229-How-to-Add-Vehicle-Fitments-for-Your-Part", 
        BASE + "115001118551-Your-Parts-Fitment",
        BASE + "35670603361037-Different-Marketplaces",
    ],
    "Parts and Fitment": [
        BASE + "28139108360205-Inherited-Part",
        BASE + "360039073611-Error-Fitment-Correction",
        BASE + "38490070236173-How-to-Delete-Fitments-for-a-Specific-Part-in-PCFitment",
        BASE + "12407054585229-Update-ASIN",
        BASE + "28822668816653-Auto-Sync-in-PCFitment",
    ],
    "Batch Upload and Download": [
        BASE + "12406713125645-Download-Templates",
        BASE + "360036247991-How-To-Use-PCFitment-Excel-Template",
        BASE + "41100889525901-How-to-Import-Parts-Using-Batch-Upload",
        BASE + "360036248371-How-to-Add-Fitments-Using-Batch-Upload",
        BASE + "36140291600909-Helpful-Function-for-Batch-Upload",
        BASE + "39965095680013-Batch-Uploading-eBay-Specific-Vehicle-Fitment",
        BASE + "46062963658381-How-to-View-Fitment-Data-After-Batch-Upload",
    ],
    "Universal Collection": [
        BASE + "20491455758221-Add-Edit-Universal-Collection",
        BASE + "20491676817805-Add-Part-into-Universal-Collection",
        BASE + "20492059135373-Add-existing-part-into-Universal-Collection-Update-collection-of-a-part",
    ],
    "Submit to Amazon": [
        BASE + "35672910870541-Amazon-Support-Resources",
    ],
    "Submit to eBay": [
        BASE + "35672925923469-eBay-Support-Resources",
    ],
    "Submit to Walmart": [
        BASE + "12407539528845-Walmart-Product-Setup",
        BASE + "9740249574797-Submit-Fitments-on-Walmart",
    ],
    "Submit to StoreAutomator": [
        BASE + "35671181058573-Submit-to-StoreAutomator",
    ],
    "Submit to SureDone": [
        BASE + "35671214822541-Submit-to-Suredone",
    ],
    "Sync with ChannelAdvisor": [
        BASE + "36377588042125-How-to-setup-channeladvisor-with-PCFitment",
    ],
    "Amazon Marketplace Guides": [
        BASE + "115001118651-Submit-ACES-to-Amazon",
        BASE + "23609211027469-Setup-AmazonConfirmedFit-on-the-Product",
        BASE + "35825733853837-Amazon-Fitment-Stripe-Not-Visible-Here-s-Why",
        BASE + "23609607828237-How-to-use-AmazonConfirmed-Fit-in-PCFitment",
        BASE + "360039073091-Amazon-Category-Selection",
        BASE + "360038681072-Amazon-PCFitment-Mapping",
        BASE + "23853480102669-Error-Code-Helps-for-AmazonConfirmedFit",
        BASE + "115001319832-Troubleshooting-Errors",
        BASE + "115001118691-How-Long-Does-It-Take-for-Fitment-to-Show-on-Amazon",
    ],
    "eBay Management": [
        BASE + "11316466305293-eBay-Subscriptions-Plan",
        BASE + "11316643816717-How-to-setup-eBay-with-PCFitment",
        BASE + "11316343045133-eBay-PartNumber-SKU-Mapping",
        BASE + "11315588106253-eBay-Fitments-in-Excel-from-PCFitment",
        BASE + "11316766570381-Submit-Compatibility-Listings-on-eBay",
        BASE + "36410081563149-eBay-Vehicle-Lookup-Viewer",
        BASE + "39964842410893-How-to-add-eBay-Specific-Vehicle-Fitment",
    ],
    "eBay to Amazon": [
        BASE + "11316125772813-Download-Fitment-from-eBay-and-Convert-it-to-Amazon-supported-format",
        BASE + "11315925580429-How-do-I-copy-fitment-from-eBay-to-the-existing-Part-Number",
    ],
    "OE to Amazon": [
        BASE + "35702188846093-Download-Fitment-from-OE-and-Convert-it-to-Amazon-supported-format",
        BASE + "35702324712589-How-do-I-copy-fitment-from-OE-Original-Equipment-to-the-existing-Part-Number",
    ],
    "VIN to Amazon": [
        BASE + "47664988082701-How-do-I-copy-fitment-from-VIN-Vehicle-Identification-Number-the-existing-Part-Number",
        BASE + "47664934096909-Download-Fitment-from-VIN-and-Convert-it-to-Amazon-Supported-Format",
    ],
    "Auto Care Database Viewer": [
        BASE + "115001705212-Auto-Care-Association-Updates",
        BASE + "115001320052-PCDB-Viewer",
        BASE + "115001313751-VCDB-Viewer",
        BASE + "31443402521613-Coverage-Report-Data-Guide",
    ],
    "Plugin Integration": [
        BASE + "35670757877261-Plugin-Integration",
        BASE + "37261383530765-How-to-Add-YMM-Filter-Integration-to-Your-eCommerce-Store-Using-PCFitment",
    ],
    "VIN Search": [
        BASE + "35670795634061-VIN-Search-Solution-for-Automotive-eCommerce",
    ],
    "Helpful Articles": [
        BASE + "35857655646605-How-do-I-copy-fitment-from-PCFitment-to-the-existing-Part-Number",
        BASE + "42907184234509-PCFitment-User-Guide-Custom-Fitment",
        BASE + "42906925322765-Automating-Amazon-Product-Reviews-with-PCFitment",
        BASE + "41468816912525-Managing-Overlap-Fitments",
        BASE + "37985033414157-Introducing-PCCoins-Get-Rewarded-for-Every-Subscription",
        BASE + "115001695511-Coming-from-another-provider",
        BASE + "41071243381773-Amazon-Walmart-Part-Mapping-Function-in-PCFitment",
        BASE + "36112569832973-What-is-the-Resubmit-Fitments-Feature",
        BASE + "35823588036877-Forgot-Password",
        BASE + "35673653590157-Use-of-Brand-Code",
        BASE + "115001118251-FAQ",
    ],
    "Billing and Pricing": [
        BASE + "35794715143437-After-Your-Trial-Ends-Choose-the-Right-PCFitment-Subscription-Plan",
        BASE + "360035048172-How-to-update-change-your-credit-card-information",
        BASE + "115001118271-Pricing",
    ],
}

# Flat list (kept for compatibility)
ARTICLE_URLS = [url for urls in CATEGORIES.values() for url in urls]


def title_from_url(url: str) -> str:
    slug = url.rstrip("/").split("/")[-1]
    parts = slug.split("-", 1)
    return (parts[1] if len(parts) > 1 else slug).replace("-", " ")


def load_article(url: str, category: str = ""):
    """Load a single article body, falling back to the full page."""

    loader = WebBaseLoader(
        web_paths=[url],
        bs_kwargs={"parse_only": bs4.SoupStrainer(class_="article-body")},
    )

    documents = loader.load()

    if not documents or not documents[0].page_content.strip():
        print(f"article-body not found for {url}, loading full page instead...")
        documents = WebBaseLoader(web_paths=[url]).load()

    for doc in documents:
        doc.metadata["source"] = url
        doc.metadata["category"] = category
        doc.metadata["title"] = title_from_url(url)

    return documents
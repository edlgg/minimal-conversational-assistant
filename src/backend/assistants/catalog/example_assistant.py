from pydantic import BaseModel, Field
from ast import literal_eval
import pandas as pd
import base64
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta, timezone

from backend.assistants.base_assistant import BaseAssistant
from backend.assistants.assistant_types import AssistantState

def make_line_chart(
    df: pd.DataFrame,
    title: str,
    x_title: str,
    y_title: str,
    height: int = 600,
    width: int = 1000,
    as_image: bool = False,
):
    fig = go.Figure()
    for column in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[column], mode="lines", name=column))

    fig.update_xaxes(
        title=x_title,
    )

    fig.update_yaxes(title=y_title)

    # Disable interactivity
    fig.update_layout(
        autosize=True,
        title=f"{title}",
        # Set chart background color to transparent
        # plot_bgcolor="rgba(0, 0, 0, 0)",
        # paper_bgcolor="rgba(0, 0, 0, 0)",
        # Set the font color to white
        # font=dict(color="white"),
        # width=width,
        # height=height,
        margin=dict(l=0, r=0, t=40, b=20),
    )
    if as_image:
        image_data = fig.to_image(format="png", width=width, height=height, scale=3)
        base64_image = base64.b64encode(image_data).decode("utf-8")
        html = f'<img src="data:image/png;base64,{base64_image}">'
        return html

    html = fig.to_html(
        include_plotlyjs=False,
        full_html=False,
    )

    return html

class PriceChartPrinter(BaseModel):
    """Tool to start workflow to make a reservation.
    Use when the user wants to make a new reservation or when the user has selected a restaurant.
    """
    symbol: str = Field(
        ..., description="The stock symbol of the company to get the price chart for."
    )

class FinanceSupportAssistant(BaseAssistant):
    name: str = 'Finance Support Assistant'
    tools: list = [
        PriceChartPrinter,
    ]

    async def generate_response(self, state: AssistantState) -> str:
        """Generate response by querying the model or processing input. Defined in subclass."""
        instruction = "Respond to user input."
        response = await self.llm_query(instruction, state.messages, self.config.llm_model)
        function_call = response.additional_kwargs.get('function_call')
        tool_calls = response.additional_kwargs.get('tool_calls')
        response_text = str(response.content)
        if function_call or tool_calls:
            response.content = "tool_call"
            response_text = await self.handle_tool_call(response, function_call, tool_calls)

        # response_text = response_text.replace("\n\n\n", "\n")
        return response_text

    async def handle_tool_call(self, response, function_call, tool_calls) -> str:
        """Handle tool call."""
        arguments_str = function_call["arguments"]
        arguments = literal_eval(arguments_str)
        if function_call["name"] == PriceChartPrinter.__name__:
            return await self.price_chart_printer(arguments["symbol"])
        raise NotImplementedError(f"Function call {function_call} not implemented.")

    async def price_chart_printer(self, symbol: str) -> str:
        """Handle PriceChartPrinter tool call."""

        # apple = yf.Ticker("AAPL")
        # print(apple.info)
        # print("apple recommendations", apple.recommendations)
        # print("apple recommendations summary", apple.recommendations_summary)
        # print("apple analyst price targets", apple.analyst_price_targets)
        # print("apple dividends", apple.dividends)
        # print("major holders", apple.major_holders)
        # print("institutional holders", apple.institutional_holders)
        # print("balance sheet", apple.balance_sheet)
        # print("cashflow", apple.cashflow)
        now = datetime.now(timezone.utc)
        start_date = now - timedelta(days=365)
        data = yf.download(symbol.upper(), start=start_date, end=now)
        data.columns = [col.lower() for col, symbol in data.columns]

        # fig = go.Figure()
        # fig.add_trace(go.Scatter(x=data.index, y=data['adj close'], mode='lines', name='Adjusted Close'))
        # fig.update_layout(
        #     title=f"{symbol.upper()} Stock Price Chart",
        #     xaxis_title="Date",
        #     yaxis_title="Adjusted Close Price (USD)",
        #     template="plotly_white",
        # )

        # html = fig.to_html(include_plotlyjs=False, full_html=False)
        # html = fig.to_html()
        data = data[["high", "low", "close"]]
        chart = make_line_chart(
            df=data,
            title=f"{symbol.upper()} Stock Price Chart",
            x_title="Date",
            y_title="Adjusted Close Price (USD)",
            height=500,
            width=1000,
        )

        return chart
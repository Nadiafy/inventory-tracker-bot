import os
import io
import json
import asyncio
from typing import List, Optional, Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from aiohttp import web

from aiogram import Bot, Dispatcher, F, types
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from google import genai
from google.genai import types as genai_types
import gspread

# Load local .env if present
load_dotenv()

# ==========================================
# 1. PYDANTIC SCHEMAS FOR GEMINI EXTRACTION
# ==========================================

class YarnRecord(BaseModel):
    date: str = Field(description="تاریخ سند یا بارنامه به شمسی مثل 1405/07/19")
    yarn_code: str = Field(description="نوع یا نمره/کد نخ مثل نخ 6.20 بوبین یا نخ 1/30 آسیاب یا نخ 50/2")
    sender: str = Field(description="فرستنده یا مبدا مثل سهیل نسج یا نساجی طره یزد")
    acceptor: str = Field(description="گیرنده یا مقصد مثل نساجی مازندران یا عسگرانی")
    weight_kg: float = Field(description="وزن خالص به کیلوگرم (اولویت با دست‌نویس انبار نسبت به بارنامه چاپی)")

class FabricRow(BaseModel):
    sender: str = Field(description="فرستنده مثل نساجی مازندران یا مرشدی")
    fabric_code: str = Field(description="کد کالیته یا نوع پارچه مثل 228417/20/H یا روسری آهاری")
    date: str = Field(description="تاریخ حواله یا فاکتور به شمسی")
    value: float = Field(description="مقدار / متراژ (یا ارزش) هر ردیف")

class ExtractionResult(BaseModel):
    category: Literal["yarn", "fabric", "other"] = Field(
        description="تعیین اینکه سند مربوط به نخ است یا پارچه یا سایر موارد (مثل فیش بانکی)"
    )
    yarn_data: Optional[List[YarnRecord]] = Field(default=None, description="پر شود اگر سند مربوط به نخ است")
    fabric_data: Optional[List[FabricRow]] = Field(default=None, description="لیست تمام ردیف‌های پارچه اگر سند مربوط به پارچه است")
    summary_persian: str = Field(description="خلاصه کوتاه به فارسی برای نمایش در تلگرام جهت تایید")

# ==========================================
# 2. BOT & CLIENTS SETUP
# ==========================================

BOT_TOKEN = os.getenv("BOT_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not BOT_TOKEN or not GEMINI_API_KEY:
    raise ValueError("❌ Missing BOT_TOKEN or GEMINI_API_KEY environment variables!")

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# Google Sheets Connection (Render Env Var OR local service_account.json)
google_creds_json = os.getenv("GOOGLE_CREDENTIALS_JSON")
if google_creds_json:
    # Used when running on Render
    creds_dict = json.loads(google_creds_json)
    gc = gspread.service_account_from_dict(creds_dict)
else:
    # Used when running locally on Mac
    gc = gspread.service_account(filename="service_account.json")

yarn_sheet = gc.open("Yarn_Tracking").sheet1
fabric_sheet = gc.open("Fabric_Tracking").sheet1

# Auto-create headers if Row 1 is empty
if not yarn_sheet.row_values(1):
    yarn_sheet.append_row(["تاریخ (Date)", "نوع / کد نخ (Yarn Code)", "فرستنده (Sender)", "گیرنده (Acceptor)", "وزن خالص - کیلو (Weight kg)"])
if not fabric_sheet.row_values(1):
    fabric_sheet.append_row(["فرستنده (Sender)", "کد پارچه (Fabric Code)", "تاریخ (Date)", "مقدار / متراژ (Value)"])

# In-memory queues:
photo_buffers: dict[tuple[int, int], list[bytes]] = {}
pending_approvals: dict[str, ExtractionResult] = {}

SYSTEM_PROMPT = """
تو یک دستیار هوشمند حسابداری نساجی هستی.
تصاویر ارسالی (حواله انبار، بارنامه، فاکتور فروش، یا دست‌نویس) و متن توضیح کاربر را بررسی کن:
1. تشخیص بده سند مربوط به «نخ» (yarn) است یا «پارچه» (fabric). اگر فیش واریزی بانکی است «other» برگردان.
2. برای نخ (yarn): اگر هم بارنامه چاپی و هم رسید دست‌نویس وجود دارد، آنها را ترکیب کن و حتماً «وزن خالص» دقیق دست‌نویس را به عنوان وزن ثبت کن. فرستنده و گیرنده را از روی متن کاربر و سند استخراج کن.
3. برای پارچه (fabric): اگر جدول شامل چند ردیف کد کالیته مختلف است (مثل 228417/20/H و غیره)، تک‌تک ردیف‌ها را جداگانه با کد پارچه و متراژ/مقدار همان ردیف استخراج کن. اگر فاکتور کلی است (مثل حواله طاقه‌ای مرشدی)، جمع کل متراژ را ثبت کن.
4. تصاویر ممکن است ۹۰ درجه چرخیده باشند؛ آنها را به درستی بخوان و اعداد فارسی را به انگلیسی تبدیل کن.
"""

# ==========================================
# 3. HANDLERS: QUEUE PHOTOS -> TRIGGER ON TEXT
# ==========================================

@dp.message(F.photo)
async def buffer_photo(message: types.Message):
    key = (message.chat.id, message.from_user.id)
    if key not in photo_buffers:
        photo_buffers[key] = []

    photo = message.photo[-1]
    file_info = await bot.get_file(photo.file_id)
    img_stream = io.BytesIO()
    await bot.download_file(file_info.file_path, destination=img_stream)
    photo_buffers[key].append(img_stream.getvalue())

    if message.caption:
        await process_queued_images(message, message.caption)


@dp.message(F.text)
async def trigger_on_text(message: types.Message):
    key = (message.chat.id, message.from_user.id)
    text = message.text.strip()

    # Test command to verify the bot is online
    if text in ["/start", "/ping", "تست"]:
        await message.reply("✅ ربات حسابداری نخ و پارچه فعال است!\nابتدا عکس(های) حواله را بفرستید و سپس توضیح آن را بنویسید.")
        return

    # Cancel command to clear accidentally uploaded photos
    if text in ["/cancel", "لغو", "پاک کردن"]:
        if key in photo_buffers and photo_buffers[key]:
            photo_buffers.pop(key, None)
            await message.reply("🗑️ عکس‌های قبلی پاک شدند. می‌توانید عکس جدید بفرستید.")
        return

    # Only process if this user has uploaded photos waiting in the queue
    if key not in photo_buffers or len(photo_buffers[key]) == 0:
        return

    await process_queued_images(message, text)


async def process_queued_images(message: types.Message, user_text: str):
    key = (message.chat.id, message.from_user.id)
    images_bytes = photo_buffers.pop(key, [])
    if not images_bytes:
        return

    status_msg = await message.reply(f"⏳ در حال بررسی {len(images_bytes)} تصویر با توضیح: «{user_text}»...")

    contents = [SYSTEM_PROMPT, f"توضیح کاربر در گروه: {user_text}"]
    for img_b in images_bytes:
        contents.append(
            genai_types.Part.from_bytes(data=img_b, mime_type="image/jpeg")
        )

    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=genai_types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ExtractionResult,
                temperature=0.1,
            ),
        )
        result: ExtractionResult = ExtractionResult.model_validate_json(response.text)
    except Exception as e:
        await status_msg.edit_text(f"❌ خطا در خواندن تصویر: {e}")
        return

    if result.category == "other":
        await status_msg.edit_text("ℹ️ این تصویر مربوط به حواله نخ یا پارچه نیست (احتمالاً فیش بانکی است) و ثبت نشد.")
        return

    batch_id = f"{message.chat.id}_{message.message_id}"
    pending_approvals[batch_id] = result

    if result.category == "yarn" and result.yarn_data:
        preview_lines = ["🧶 تشخیص: حواله نخ\n"]
        for y in result.yarn_data:
            preview_lines.append(
                f"📅 تاریخ: {y.date}\n"
                f"🔹 نوع/کد نخ: {y.yarn_code}\n"
                f"📤 فرستنده: {y.sender}\n"
                f"📥 گیرنده: {y.acceptor}\n"
                f"⚖️ وزن خالص: {y.weight_kg:,.2f} کیلوگرم\n"
            )
    else:
        preview_lines = [f"🧵 تشخیص: حواله پارچه ({len(result.fabric_data or [])} ردیف)\n"]
        total_val = 0
        for idx, f_row in enumerate(result.fabric_data or [], 1):
            total_val += f_row.value
            preview_lines.append(
                f"{idx}. کد: {f_row.fabric_code} | مقدار: {f_row.value:,.0f} | فرستنده: {f_row.sender} ({f_row.date})"
            )
        preview_lines.append(f"\n📊 جمع کل: {total_val:,.0f}")

    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="✅ تایید و ثبت در شیت", callback_data=f"approve:{batch_id}"),
            InlineKeyboardButton(text="❌ لغو", callback_data=f"reject:{batch_id}")
        ]
    ])

    await status_msg.edit_text("\n".join(preview_lines), reply_markup=keyboard)


# ==========================================
# 4. CALLBACK HANDLER: WRITE TO GOOGLE SHEETS
# ==========================================

@dp.callback_query(F.data.startswith(("approve:", "reject:")))
async def handle_approval(callback: types.CallbackQuery):
    action, batch_id = callback.data.split(":", 1)
    result = pending_approvals.pop(batch_id, None)

    if not result:
        await callback.answer("این مورد قبلاً تعیین تکلیف شده است.", show_alert=True)
        return

    user_name = callback.from_user.full_name

    if action == "reject":
        await callback.message.edit_text(
            callback.message.text + f"\n\n❌ لغو شد توسط {user_name}"
        )
        await callback.answer("لغو شد.")
        return

    try:
        if result.category == "yarn" and result.yarn_data:
            rows_to_append = [
                [y.date, y.yarn_code, y.sender, y.acceptor, y.weight_kg]
                for y in result.yarn_data
            ]
            yarn_sheet.append_rows(rows_to_append, value_input_option="USER_ENTERED")
            sheet_name = "شیت نخ"

        elif result.category == "fabric" and result.fabric_data:
            rows_to_append = [
                [f.sender, f.fabric_code, f.date, f.value]
                for f in result.fabric_data
            ]
            fabric_sheet.append_rows(rows_to_append, value_input_option="USER_ENTERED")
            sheet_name = f"شیت پارچه ({len(rows_to_append)} ردیف)"

        await callback.message.edit_text(
            callback.message.text + f"\n\n✅ تایید شد توسط {user_name} و در {sheet_name} ثبت گردید."
        )
        await callback.answer("با موفقیت در گوگل شیت ثبت شد!")
    except Exception as e:
        pending_approvals[batch_id] = result
        await callback.answer(f"خطا در اتصال به گوگل شیت: {e}", show_alert=True)


# ==========================================
# 5. WEB SERVER FOR RENDER FREE TIER
# ==========================================

async def health_check(request):
    return web.Response(text="Bot is alive and running!")

async def start_web_server():
    app = web.Application()
    app.router.add_get("/", health_check)
    runner = web.AppRunner(app)
    await runner.setup()
    port = int(os.environ.get("PORT", 10000))
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    print(f"🌐 Health check server listening on port {port}")

async def main():
    await start_web_server()
    print("🤖 Bot is polling...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
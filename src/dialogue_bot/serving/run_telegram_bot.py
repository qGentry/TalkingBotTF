from dialogue_bot.model.model_wrapper import ModelWrapper
import logging

from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

config = {
    'model_params': {
        'vocab_path': 'vocabs/vocab.txt',
        'context_size': 1,
        'embedding_params': {
            'vocab_size': 87,
            'embedding_size': 128,
        },
        'encoder_params': {
            'hidden_size': 200,
            'context_size': 1
        },
        'decoder_params': {
            'embedding_size': 128,
            'vocab_size': 87,
            'hidden_size': 300,
            'output_vocab_size': 87,
        }
    },
    'wrapper_params': {
        'train': False,
        'batch_size': 200,
        'clip_norm': 2,
        'data_path': '../data/TlkPersonaChatRus/dialogues.tsv',
    }
}

TOKEN = ""

model = ModelWrapper(params=config)
model.restore_model('model/')
model([['привет']])
# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def start(update, context):
    update.message.reply_text('Hi!')


def help_command(update, context):
    update.message.reply_text('Help!')


def echo(update, context):
    """Echo the user message."""
    text = update.message.text
    response = model([[text]])
    update.message.reply_text(response[0][0])


def main():
    """Start the bot."""
    # Create the Updater and pass it your bot's token.
    # Make sure to set use_context=True to use the new context based callbacks
    # Post version 12 this will no longer be necessary
    updater = Updater(TOKEN, use_context=True)

    # Get the dispatcher to register handlers
    dp = updater.dispatcher

    # on different commands - answer in Telegram
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", help_command))

    # on noncommand i.e message - echo the message on Telegram
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, echo))

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()


if __name__ == '__main__':
    main()

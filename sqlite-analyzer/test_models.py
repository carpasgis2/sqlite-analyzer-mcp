
import openai

openai.api_key = "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"


try:
    # Listar modelos disponibles
    response = openai.Model.list()
    for m in response.data:
        print(m.id)
except Exception as e:
    print(f"Error listando modelos: {e}")
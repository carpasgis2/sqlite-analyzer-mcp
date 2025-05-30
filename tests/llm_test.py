from openai import OpenAI

# Se recomienda configurar la API key como una variable de entorno OPENAI_API_KEY.
# Si la tienes configurada así, puedes instanciar el cliente sin pasar la api_key:
# client = OpenAI()
# O, si necesitas pasarla explícitamente (menos recomendado para código compartido):
client = OpenAI(api_key="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")

try:
    response = client.chat.completions.create(
      model="gpt-4.5-preview",
      messages=[{"role": "user", "content": "¿Cómo estás?"}]
    )
    print(response.choices[0].message.content)
except Exception as e:
    print(f"Ocurrió un error al contactar con OpenAI: {e}")

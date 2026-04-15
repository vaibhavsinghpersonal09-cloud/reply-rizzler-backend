require('dotenv').config();
const express = require('express');
const multer = require('multer');
const cors = require('cors');
const path = require('path');
const fs = require('fs');
const axios = require('axios');
const vision = require('@google-cloud/vision');

const app = express();
const PORT = process.env.PORT || 3000;
const GOOGLE_API_KEY = process.env.GOOGLE_API_KEY;
const VISION_API_URL = GOOGLE_API_KEY
  ? `https://vision.googleapis.com/v1/images:annotate?key=${GOOGLE_API_KEY}`
  : null;

const GOOGLE_APPLICATION_CREDENTIALS_JSON = process.env.GOOGLE_APPLICATION_CREDENTIALS_JSON;
let visionClient = null;
if (GOOGLE_APPLICATION_CREDENTIALS_JSON) {
  try {
    let credsObj;
    try {
      credsObj = JSON.parse(GOOGLE_APPLICATION_CREDENTIALS_JSON);
    } catch (_) {
      const decoded = Buffer.from(GOOGLE_APPLICATION_CREDENTIALS_JSON, 'base64').toString('utf8');
      credsObj = JSON.parse(decoded);
    }

    visionClient = new vision.ImageAnnotatorClient({
      credentials: credsObj
    });
  } catch (e) {
    console.error('Failed to initialize Google Vision client from GOOGLE_APPLICATION_CREDENTIALS_JSON:', e?.message || e);
    visionClient = null;
  }
}

function detectIndianChatLanguage(text) {
  const t = (text || '').toString();
  if (!t.trim()) return 'hinglish';

  const devanagariMatches = t.match(/[\u0900-\u097F]/g) || [];
  const latinMatches = t.match(/[A-Za-z]/g) || [];
  const totalLetters = devanagariMatches.length + latinMatches.length;

  if (totalLetters === 0) return 'hinglish';

  const devanagariRatio = devanagariMatches.length / totalLetters;
  if (devanagariRatio > 0.55) return 'hindi';

  const hinglishSignals = [
    /\b(yaar|yr|bro|bhai|behen|bhen|didi|babu|jaan|ya|kyu|kyun|kya|kaise|nahi|haan|haa|ha|matlab|scene|vibe|chal|arey|arrey|hoga|nhi|nai)\b/i,
    /\b(lol|lmao|bruh|vibe|scene|sahi|mast|pagal|crazy)\b/i
  ];
  const hasHinglishSignal = hinglishSignals.some((re) => re.test(t));
  if (hasHinglishSignal) return 'hinglish';

  return 'english';
}

function pickReplyLengthHint(spicinessLevel) {
  const lvl = Number.isFinite(Number(spicinessLevel)) ? Number(spicinessLevel) : 1;
  const clamped = Math.min(Math.max(lvl, 1), 3);
  if (clamped === 1) return '1 short line (max ~12 words).';
  if (clamped === 2) return '1-2 short lines (max ~20 words).';
  return '1-2 short lines (max ~22 words).';
}

function spiceStyleGuide(spicinessLevel) {
  const lvl = Number.isFinite(Number(spicinessLevel)) ? Number(spicinessLevel) : 1;
  const clamped = Math.min(Math.max(lvl, 1), 3);
  return {
    1: 'sweet, confident, low-key charming. Safe and not over-flirty.',
    2: 'playful banter, light tease, clear interest. Still classy.',
    3: 'high-rizz, bold flirt, witty. Spicy but not vulgar, no explicit sexual content.'
  }[clamped];
}

function buildReplyRizzlerPrompt({ ocrText, userContext, spicinessLevel }) {
  const ocr = (ocrText || '').toString().trim();
  const ctx = (userContext || '').toString().trim();
  const combined = [ocr, ctx].filter(Boolean).join('\n');
  const lang = detectIndianChatLanguage(combined);
  const lengthHint = pickReplyLengthHint(spicinessLevel);
  const spiceGuide = spiceStyleGuide(spicinessLevel);

  const languageRule =
    lang === 'hindi'
      ? 'Write in Hindi (Devanagari).'
      : lang === 'hinglish'
        ? 'Write in Hinglish (Hindi in Latin letters + a bit of English, like Indian Gen Z texting).'
        : 'Write in English, but keep it India-friendly (natural texting).';

  return {
    lang,
    messages: [
      {
        role: 'system',
        content: [
          'You are Reply Rizzler — you generate ONE reply message for Indian Gen Z texting.',
          'Never mention AI, OCR, analysis, or policy.',
          'No cringe, no try-hard pickup lines, no overuse of emojis.',
          'No slurs. No explicit sexual content. Keep it respectful.',
          'Do NOT add quotes. Output ONLY the final reply text.'
        ].join('\n')
      },
      {
        role: 'user',
        content: [
          `Goal: craft ONE reply to the last message in this chat screenshot text.`,
          `Language: ${languageRule}`,
          `Vibe: ${spiceGuide}`,
          `Length: ${lengthHint}`,
          'Important: Match the chat’s vibe (formal/informal), and mirror any slang level naturally.',
          ctx ? `Extra context from user (may be important):\n${ctx}` : null,
          `Chat text (OCR):\n${ocr || '(empty)'}`
        ].filter(Boolean).join('\n\n')
      }
    ]
  };
}

function buildGlowUpPrompt({ message, tone, includeEmojis }) {
  const msg = (message || '').toString().trim();
  const toneStr = (tone || '').toString().trim();
  const lang = detectIndianChatLanguage(msg);

  const languageRule =
    lang === 'hindi'
      ? 'Write in Hindi (Devanagari).'
      : lang === 'hinglish'
        ? 'Write in Hinglish (Hindi in Latin letters + a bit of English, like Indian Gen Z texting).'
        : 'Write in English, but keep it India-friendly (natural texting).';

  const emojiRule = includeEmojis
    ? 'You may add 1-3 emojis total if they fit naturally (not on every word).'
    : 'Do not add any emojis.';

  const toneRule = toneStr
    ? `Tone: ${toneStr} (keep it natural for Indian Gen Z texting).`
    : 'Tone: neutral and friendly (Indian Gen Z texting).';

  return {
    lang,
    messages: [
      {
        role: 'system',
        content: [
          'You rewrite short messages to sound better for texting in India.',
          'Never mention AI, instructions, or analysis.',
          'Keep meaning the same; do not add new facts.',
          'No cringe; keep it human.',
          'Output ONLY the improved message text. No quotes. No bullet points.'
        ].join('\n')
      },
      {
        role: 'user',
        content: [
          `Task: Improve this message for sending on Insta/WhatsApp/dating apps.`,
          `Language: ${languageRule}`,
          toneRule,
          emojiRule,
          'Keep it concise and smooth.',
          `Original message:\n${msg}`
        ].join('\n\n')
      }
    ]
  };
}

function sanitizeSingleTextReply(raw) {
  if (!raw || typeof raw !== 'string') return '';

  let s = raw.trim();

  // Remove code fences if the model accidentally returns them
  s = s.replace(/^```[a-zA-Z]*\s*/g, '').replace(/```\s*$/g, '').trim();

  // Remove surrounding quotes
  if ((s.startsWith('"') && s.endsWith('"')) || (s.startsWith('\'') && s.endsWith('\''))) {
    s = s.slice(1, -1).trim();
  }

  // If multiple lines are returned, keep the first meaningful 1-2 lines
  const lines = s
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter(Boolean);

  if (lines.length === 0) return '';
  if (lines.length === 1) return lines[0];

  // Keep up to 2 lines max (still feels like a single chat message)
  return `${lines[0]}\n${lines[1]}`.trim();
}

console.log('Vision auth mode:', visionClient ? 'service_account' : (VISION_API_URL ? 'api_key' : 'not_configured'));

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const OPENAI_MODEL = process.env.OPENAI_MODEL || 'gpt-4o-mini';
const OPENAI_MODEL_FALLBACKS = (process.env.OPENAI_MODEL_FALLBACKS || 'gpt-4o-mini')
  .split(',')
  .map(s => s.trim())
  .filter(Boolean);

async function callOpenAIChatCompletionsWithFallback(payload, { preferredModel }) {
  if (!OPENAI_API_KEY) {
    throw new Error('OPENAI_API_KEY is not configured');
  }

  const modelsToTry = [preferredModel, ...OPENAI_MODEL_FALLBACKS]
    .filter(Boolean)
    .filter((m, i, arr) => arr.indexOf(m) === i);

  let lastError = null;
  for (const model of modelsToTry) {
    try {
      const response = await axios.post(
        'https://api.openai.com/v1/chat/completions',
        { ...payload, model },
        {
          headers: {
            Authorization: `Bearer ${OPENAI_API_KEY}`,
            'Content-Type': 'application/json'
          },
          timeout: 25000
        }
      );
      return response;
    } catch (err) {
      lastError = err;
      const status = err?.response?.status;
      const msg = err?.response?.data?.error?.message || err?.message || '';
      const isModelAccessDenied = status === 403 && /does not have access to model/i.test(msg);
      if (isModelAccessDenied) {
        continue;
      }
      throw err;
    }
  }

  throw lastError || new Error('OpenAI request failed');
}

// Middleware
app.use(cors());
app.use(express.json());

// Create uploads directory if it doesn't exist
const uploadDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir, { recursive: true });
}

// Configure multer for file uploads
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, uploadDir);
  },
  filename: function (req, file, cb) {
    cb(null, Date.now() + path.extname(file.originalname));
  }
});

// File filter function
const fileFilter = function(req, file, cb) {
  const filetypes = /jpeg|jpg|png|gif/;
  const mimetype = filetypes.test(file.mimetype);
  const extname = filetypes.test(path.extname(file.originalname).toLowerCase());
  
  if (mimetype && extname) {
    return cb(null, true);
  }
  cb(new Error('Only image files are allowed (JPEG, JPG, PNG, GIF)'));
};

const upload = multer({ 
  storage: storage,
  limits: { 
    fileSize: 10 * 1024 * 1024, // 10MB limit
    files: 1
  },
  fileFilter: fileFilter
});

// Function to analyze image using Google Cloud Vision API
async function analyzeImageWithVision(imageBuffer) {
  try {
    const base64Image = imageBuffer.toString('base64');

    if (visionClient) {
      const [result] = await visionClient.annotateImage({
        image: { content: base64Image },
        features: [{ type: 'TEXT_DETECTION', maxResults: 10 }]
      });
      return { responses: [result] };
    }

    if (!VISION_API_URL) {
      throw new Error('Google Vision credentials are not configured (set GOOGLE_APPLICATION_CREDENTIALS_JSON or GOOGLE_API_KEY)');
    }

    const request = {
      requests: [
        {
          image: {
            content: base64Image
          },
          features: [
            {
              type: 'TEXT_DETECTION',
              maxResults: 10
            }
          ]
        }
      ]
    };

    const response = await axios.post(VISION_API_URL, request, {
      timeout: 25000 // 25 second timeout
    });
    return response.data;
  } catch (error) {
    const status = error?.response?.status;
    const data = error?.response?.data;
    console.error('Error analyzing image with Vision API:', {
      message: error?.message,
      status,
      data
    });
    throw error;
  }
}

async function generateOpenAIResponse({ text, spicinessLevel, context }) {
  if (!OPENAI_API_KEY) {
    throw new Error('OPENAI_API_KEY is not configured');
  }

  const userContext = (context || '').toString().trim();
  const ocrText = (text || '').toString().trim();
  const prompt = buildReplyRizzlerPrompt({
    ocrText,
    userContext,
    spicinessLevel
  });

  const payload = {
    messages: prompt.messages,
    temperature: 0.65,
    max_tokens: 120
  };

  const response = await callOpenAIChatCompletionsWithFallback(payload, { preferredModel: OPENAI_MODEL });

  const content = response?.data?.choices?.[0]?.message?.content;
  if (!content || typeof content !== 'string') {
    throw new Error('No response from OpenAI');
  }
  const cleaned = sanitizeSingleTextReply(content);
  if (!cleaned) {
    throw new Error('Empty response from OpenAI');
  }
  return cleaned;
}

// Routes
app.get('/', (req, res) => {
  res.send('Hangly Backend is running!');
});

app.post('/api/glowup', async (req, res) => {
  try {
    const { message, tone, include_emojis, count } = req.body || {};

    const msg = (message || '').toString().trim();
    const toneStr = (tone || '').toString().trim();
    const includeEmojis = include_emojis === true || include_emojis === 'true' || include_emojis === 1 || include_emojis === '1';
    const requestedCountRaw = Number(count);
    const requestedCount = Number.isFinite(requestedCountRaw) ? requestedCountRaw : 1;
    const n = Math.max(1, Math.min(3, Math.floor(requestedCount)));

    if (!msg) {
      return res.status(400).json({
        success: false,
        error: 'Message is required',
        message: 'Message is required'
      });
    }

    if (!OPENAI_API_KEY) {
      return res.status(500).json({
        success: false,
        error: 'OPENAI_API_KEY is not configured',
        message: 'OPENAI_API_KEY is not configured'
      });
    }

    const glowPrompt = buildGlowUpPrompt({
      message: msg,
      tone: toneStr,
      includeEmojis
    });

    const payload = {
      messages: glowPrompt.messages,
      temperature: 0.55,
      max_tokens: 200,
      n
    };

    const response = await callOpenAIChatCompletionsWithFallback(payload, { preferredModel: OPENAI_MODEL });

    const choices = response?.data?.choices;
    if (!Array.isArray(choices) || choices.length === 0) {
      throw new Error('No response from OpenAI');
    }

    const improvedMessages = choices
      .map((c) => c?.message?.content)
      .filter((c) => typeof c === 'string')
      .map((c) => sanitizeSingleTextReply(c))
      .filter(Boolean);

    if (improvedMessages.length === 0) {
      throw new Error('Empty response from OpenAI');
    }

    return res.json({
      success: true,
      improved_message: improvedMessages[0],
      improved_messages: improvedMessages.slice(0, n),
      tone: toneStr,
      include_emojis: includeEmojis,
      count: n
    });
  } catch (error) {
    const upstreamStatus = error?.response?.status;
    const upstreamData = error?.response?.data;
    const upstreamMessage =
      upstreamData?.error?.message ||
      upstreamData?.error?.type ||
      upstreamData?.error?.code ||
      (typeof upstreamData === 'string' ? upstreamData : null);

    console.error('Unexpected error in /api/glowup:', {
      message: error?.message,
      upstream_status: upstreamStatus,
      upstream_message: upstreamMessage,
      upstream_data: upstreamData
    });
    return res.status(500).json({
      success: false,
      error: 'Failed to glow up message',
      message: upstreamMessage || error?.message || 'Failed to glow up message',
      details: upstreamData || error?.message,
      upstream_status: upstreamStatus
    });
  }
});

// Analyze image and generate response
app.post('/api/analyze', (req, res) => {
  upload.single('image')(req, res, async (err) => {
    try {
      if (err instanceof multer.MulterError) {
        console.error('Multer error:', err);
        return res.status(400).json({ 
          error: 'File upload error',
          details: err.message 
        });
      } else if (err) {
        console.error('Upload error:', err);
        return res.status(400).json({ 
          error: 'File upload failed',
          details: err.message 
        });
      }

      const { spiciness_level, context } = req.body;
      const imagePath = req.file?.path;

      console.log('Received request with file:', req.file);
      console.log('Spiciness level:', spiciness_level);
      console.log('Context:', context);

      if (!imagePath) {
        return res.status(400).json({ 
          error: 'No image file provided',
          receivedFiles: req.files,
          body: req.body
        });
      }

      try {
        // Read image file
        console.log('Reading image file from:', imagePath);
        const imageBuffer = fs.readFileSync(imagePath);
        
        // Analyze image with Google Cloud Vision
        console.log('Analyzing image with Vision API...');
        const visionResult = await analyzeImageWithVision(imageBuffer);
        
        // Extract labels from vision result
        const labels = visionResult.responses[0]?.labelAnnotations?.map(label => label.description) || [];
        const text = visionResult.responses[0]?.textAnnotations?.[0]?.description || '';
        
        console.log('Detected labels:', labels);
        console.log('Detected text:', text);
        
        // Generate response based on detected content
        const rawLevel = parseInt(spiciness_level || '1');
        const spicinessInt = Number.isFinite(rawLevel) ? Math.min(Math.max(rawLevel, 1), 3) : 1;
        let response;
        if (!text || text.trim().length === 0) {
          throw new Error('No readable chat text detected in the image. Try a clearer screenshot (crop, increase brightness, avoid blur).');
        }

        try {
          response = await generateOpenAIResponse({
            text,
            spicinessLevel: spicinessInt,
            context
          });
        } catch (openAiError) {
          console.error('OpenAI error in /api/analyze:', {
            message: openAiError?.message,
            upstream_status: openAiError?.response?.status,
            upstream_data: openAiError?.response?.data
          });
          throw new Error('AI is temporarily unavailable. Please try again in a few seconds.');
        }
        console.log('Generated response:', response);

        // Clean up the uploaded file
        try {
          if (fs.existsSync(imagePath)) {
            fs.unlinkSync(imagePath);
            console.log('Temporary file deleted:', imagePath);
          }
        } catch (cleanupError) {
          console.error('Error cleaning up file:', cleanupError);
        }

        // Send the response
        res.json({
          success: true,
          response: response,
          spiciness_level: spicinessInt,
          detected_labels: labels,
          detected_text: text
        });

      } catch (processError) {
        console.error('Error processing image:', processError);
        // Clean up the uploaded file in case of error
        if (imagePath && fs.existsSync(imagePath)) {
          fs.unlinkSync(imagePath);
        }

        const upstreamStatus = processError?.response?.status;
        const upstreamData = processError?.response?.data;
        const upstreamMessage =
          upstreamData?.error?.message ||
          upstreamData?.error?.status ||
          (typeof upstreamData === 'string' ? upstreamData : null);
        
        const statusCode = /No readable chat text detected/i.test(processError?.message || '') ? 400 : 500;
        return res.status(statusCode).json({ 
          error: 'Failed to process image',
          message: upstreamMessage || processError.message,
          details: upstreamData || processError.message,
          upstream_status: upstreamStatus,
          stack: processError.stack
        });
      }
    } catch (error) {
      console.error('Unexpected error in /api/analyze:', error);
      res.status(500).json({ 
        error: 'Internal server error',
        message: error.message,
        details: error.message,
        stack: process.env.NODE_ENV === 'development' ? error.stack : undefined
      });
    }
  });
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({ 
    error: 'Something went wrong!',
    message: err.message 
  });
});

// Start server
app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});

import { useEffect, useRef, useState } from "react";

const MAX_INPUT_CHARS = 4000; // matches guardrails.max_input_chars in the assistant configs
const MAX_TEXTAREA_PX = 200;

interface ComposerProps {
  disabled: boolean;
  placeholder: string;
  onSend: (text: string) => void;
}

export function Composer({ disabled, placeholder, onSend }: ComposerProps) {
  const [value, setValue] = useState("");
  const ref = useRef<HTMLTextAreaElement | null>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, MAX_TEXTAREA_PX)}px`;
  }, [value]);

  useEffect(() => {
    if (!disabled) ref.current?.focus();
  }, [disabled]);

  const canSend = !disabled && value.trim().length > 0;

  function submit(): void {
    if (!canSend) return;
    onSend(value.trim());
    setValue("");
  }

  return (
    <form
      className="composer"
      onSubmit={(e) => {
        e.preventDefault();
        submit();
      }}
    >
      <textarea
        ref={ref}
        rows={1}
        value={value}
        maxLength={MAX_INPUT_CHARS}
        placeholder={placeholder}
        aria-label="Message"
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
            e.preventDefault();
            submit();
          }
        }}
      />
      <button type="submit" className="send-btn" disabled={!canSend} aria-label="Send message">
        <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
          <path d="M12 19V5M5 12l7-7 7 7" />
        </svg>
      </button>
    </form>
  );
}
